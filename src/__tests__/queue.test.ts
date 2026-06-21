import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AIOperationQueue, AIOperationCancelledError, isAIAbortError } from '../queue.js';
import type { AITaskContext } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Wraps a deferred promise so tests can control when a task resolves. */
function deferred<T = void>() {
    let resolve!: (v: T) => void;
    let reject!: (e: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

/** Yield one tick so setTimeout(0) callbacks in the queue can fire. */
const tick = () => new Promise<void>((r) => setTimeout(r, 0));

/** Yield enough ticks for the queue to process start + finish cycles. */
const flushQueue = async (ticks = 5) => {
    for (let i = 0; i < ticks; i++) await tick();
};

/**
 * A deferred that rejects when signal is aborted — mimics fetch() behaviour.
 * Use this in tests that need to verify running-op cancellation.
 */
function signalAwareDeferred<T = void>(signal: AbortSignal) {
    const d = deferred<T>();
    signal.addEventListener('abort', () => d.reject(signal.reason), { once: true });
    return d;
}

function makeQueue(opts?: { base?: number; burst?: number; slowOp?: number; threshold?: number }) {
    return new AIOperationQueue(
        { base: opts?.base ?? 3, burst: opts?.burst ?? 2, slowOp: opts?.slowOp },
        opts?.threshold ?? 90,
    );
}

/** Simple task that resolves immediately with a value. */
const immediate = <T>(value: T) => (_ctx: AITaskContext) => Promise.resolve(value);

// ---------------------------------------------------------------------------
// isAIAbortError
// ---------------------------------------------------------------------------

describe('isAIAbortError', () => {
    it('returns true for AIOperationCancelledError', () => {
        expect(isAIAbortError(new AIOperationCancelledError())).toBe(true);
    });

    it('returns true for DOMException AbortError', () => {
        const err = new Error('aborted');
        err.name = 'AbortError';
        expect(isAIAbortError(err)).toBe(true);
    });

    it('returns true for error with "aborted" in message', () => {
        expect(isAIAbortError(new Error('operation aborted'))).toBe(true);
    });

    it('returns false for regular errors', () => {
        expect(isAIAbortError(new Error('network error'))).toBe(false);
    });

    it('returns false for non-errors', () => {
        expect(isAIAbortError('some string')).toBe(false);
        expect(isAIAbortError(null)).toBe(false);
        expect(isAIAbortError(42)).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// Basic enqueue / resolve / reject
// ---------------------------------------------------------------------------

describe('basic enqueue', () => {
    it('resolves with the task return value', async () => {
        const q = makeQueue();
        const result = await q.enqueue('test', immediate(42));
        expect(result).toBe(42);
    });

    it('rejects when the task throws', async () => {
        const q = makeQueue();
        await expect(
            q.enqueue('test', async () => { throw new Error('boom'); }),
        ).rejects.toThrow('boom');
    });

    it('increments completedOperations on success', async () => {
        const q = makeQueue();
        await q.enqueue('t1', immediate(1));
        await q.enqueue('t2', immediate(2));
        expect(q.getStats().completedOperations).toBe(2);
    });

    it('increments failedOperations on task error', async () => {
        const q = makeQueue();
        await q.enqueue('t', async () => { throw new Error('x'); }).catch(() => {});
        expect(q.getStats().failedOperations).toBe(1);
    });

    it('passes an AbortSignal to the task', async () => {
        const q = makeQueue();
        let receivedSignal: AbortSignal | undefined;
        await q.enqueue('t', async (ctx) => { receivedSignal = ctx.signal; });
        expect(receivedSignal).toBeInstanceOf(AbortSignal);
    });

    it('runs tasks concurrently up to base limit', async () => {
        const q = makeQueue({ base: 2, burst: 0 });
        const d1 = deferred();
        const d2 = deferred();
        const d3 = deferred();

        const p1 = q.enqueue('t1', () => d1.promise);
        const p2 = q.enqueue('t2', () => d2.promise);
        const p3 = q.enqueue('t3', () => d3.promise);

        await tick();
        // Only 2 should be running (base=2); 3rd is queued
        expect(q.getStats().runningOperations).toBe(2);
        expect(q.getStats().pendingOperations).toBe(1);

        d1.resolve();
        await p1;
        await tick();
        // Now t3 should have started
        expect(q.getStats().runningOperations).toBe(2);

        d2.resolve();
        d3.resolve();
        await Promise.all([p2, p3]);
    });
});

// ---------------------------------------------------------------------------
// Priority ordering
// ---------------------------------------------------------------------------

describe('priority ordering', () => {
    it('runs higher-priority ops before lower-priority ones', async () => {
        // Use base=1 so ops queue behind the first one
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const order: string[] = [];

        // Fill the single slot
        const p0 = q.enqueue('blocker', () => blocker.promise);

        await tick();

        // Queue two ops while slot is full
        const pLow = q.enqueue('low', async () => { order.push('low'); }, { priority: 10 });
        const pHigh = q.enqueue('high', async () => { order.push('high'); }, { priority: 50 });

        // Release the blocker
        blocker.resolve();
        await p0;
        await Promise.all([pLow, pHigh]);

        expect(order).toEqual(['high', 'low']);
    });

    it('uses FIFO within the same priority', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const order: string[] = [];

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        const p1 = q.enqueue('first', async () => { order.push('first'); }, { priority: 5 });
        const p2 = q.enqueue('second', async () => { order.push('second'); }, { priority: 5 });

        blocker.resolve();
        await p0;
        await Promise.all([p1, p2]);

        expect(order).toEqual(['first', 'second']);
    });
});

// ---------------------------------------------------------------------------
// Dedupe — pending
// ---------------------------------------------------------------------------

describe('dedupe (pending)', () => {
    it('replaces a pending task with the same dedupeKey', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const results: number[] = [];

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        const p1 = q.enqueue('task', async () => { results.push(1); return 1; }, { dedupeKey: 'dk' });
        const p2 = q.enqueue('task', async () => { results.push(2); return 2; }, { dedupeKey: 'dk' });

        blocker.resolve();
        await p0;

        const [r1, r2] = await Promise.all([p1, p2]);

        // Only one execution (the replaced task), both promises resolve with same value
        expect(results).toHaveLength(1);
        expect(r1).toBe(r2);
    });

    it('bumps priority when re-enqueuing a pending key', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const order: string[] = [];

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        // Enqueue 'a' at priority 5, then 'b' at priority 20, then bump 'a' to 30
        const pA = q.enqueue('a', async () => { order.push('a'); }, { dedupeKey: 'a', priority: 5 });
        const pB = q.enqueue('b', async () => { order.push('b'); }, { dedupeKey: 'b', priority: 20 });
        // Bump 'a' — this should raise its priority above 'b'
        q.enqueue('a-bump', async () => { order.push('a-bumped'); }, { dedupeKey: 'a', priority: 30 });

        blocker.resolve();
        await p0;
        await Promise.all([pA, pB]);

        // a (bumped) should run before b
        expect(order[0]).toBe('a-bumped');
    });
});

// ---------------------------------------------------------------------------
// Dedupe — running
// ---------------------------------------------------------------------------

describe('dedupe (running)', () => {
    it('attaches to a running operation with the same dedupeKey', async () => {
        const q = makeQueue();
        const d = deferred<number>();
        let runCount = 0;

        const p1 = q.enqueue('task', async () => { runCount++; return d.promise; }, { dedupeKey: 'dk' });
        await tick();
        // Task is now running; enqueue again with same key
        const p2 = q.enqueue('task', async () => { runCount++; return d.promise; }, { dedupeKey: 'dk' });

        d.resolve(99);
        const [r1, r2] = await Promise.all([p1, p2]);

        // Should only have run once
        expect(runCount).toBe(1);
        expect(r1).toBe(99);
        expect(r2).toBe(99);
    });
});

// ---------------------------------------------------------------------------
// restart
// ---------------------------------------------------------------------------

describe('restart', () => {
    it('aborts a running operation and starts fresh', async () => {
        const q = makeQueue();
        const d2 = deferred<string>();
        let secondStarted = false;

        // Task 1: signal-aware so it rejects when aborted.
        // Attach .catch() immediately so Node doesn't flag it as unhandled
        // before our assertion runs.
        const p1 = q.enqueue('task', async (ctx) => {
            const d1 = signalAwareDeferred<string>(ctx.signal);
            return d1.promise;
        }, { dedupeKey: 'dk' });
        p1.catch(() => {});
        await tick();

        const p2 = q.enqueue('task', async (_ctx) => {
            secondStarted = true;
            return d2.promise;
        }, { dedupeKey: 'dk', restart: true });

        await tick();
        expect(secondStarted).toBe(true);
        expect(q.getStats().cancelledOperations).toBeGreaterThanOrEqual(1);

        await expect(p1).rejects.toSatisfy(isAIAbortError);

        d2.resolve('fresh');
        expect(await p2).toBe('fresh');
    });
});

// ---------------------------------------------------------------------------
// isStillValid
// ---------------------------------------------------------------------------

describe('isStillValid', () => {
    it('skips the task if isStillValid returns false when run begins', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        let taskRan = false;
        const p1 = q.enqueue(
            'task',
            async () => { taskRan = true; },
            { isStillValid: () => false },
        );

        blocker.resolve();
        await p0;
        const result = await p1;

        expect(taskRan).toBe(false);
        expect(result).toBeUndefined();
        expect(q.getStats().skippedOperations).toBe(1);
    });

    it('runs the task if isStillValid returns true', async () => {
        const q = makeQueue();
        let ran = false;
        await q.enqueue('task', async () => { ran = true; }, { isStillValid: () => true });
        expect(ran).toBe(true);
    });
});

// ---------------------------------------------------------------------------
// cancelGroup
// ---------------------------------------------------------------------------

describe('cancelGroup', () => {
    it('rejects queued operations in the group', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        const pA = q.enqueue('a', immediate(1), { group: 'g1' });
        const pB = q.enqueue('b', immediate(2), { group: 'g1' });
        const pC = q.enqueue('c', immediate(3), { group: 'g2' }); // different group

        q.cancelGroup('g1');
        blocker.resolve();
        await p0;

        await expect(pA).rejects.toSatisfy(isAIAbortError);
        await expect(pB).rejects.toSatisfy(isAIAbortError);

        // c should still run fine
        expect(await pC).toBe(3);
        expect(q.getStats().cancelledOperations).toBe(2);
    });

    it('aborts running operations in the group', async () => {
        const q = makeQueue();
        let aborted = false;

        const p = q.enqueue('task', async (ctx) => {
            const d = signalAwareDeferred(ctx.signal);
            ctx.signal.addEventListener('abort', () => { aborted = true; });
            return d.promise;
        }, { group: 'g1' });
        p.catch(() => {});

        await tick();
        q.cancelGroup('g1');
        await tick();

        expect(aborted).toBe(true);
        await expect(p).rejects.toSatisfy(isAIAbortError);
    });

    it('clears focusedGroup when cancelling it', async () => {
        const q = makeQueue();
        q.setFocusedGroup('g1');
        q.cancelGroup('g1');
        // After cancel, enqueueing something for g1 should not get burst priority
        // (just verifying no error thrown)
        await q.enqueue('t', immediate(1), { group: 'g1' });
    });
});

// ---------------------------------------------------------------------------
// cancelKey
// ---------------------------------------------------------------------------

describe('cancelKey', () => {
    it('rejects a queued operation by dedupeKey', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        const p = q.enqueue('task', immediate(1), { dedupeKey: 'myKey' });
        q.cancelKey('myKey');

        blocker.resolve();
        await p0;
        await expect(p).rejects.toSatisfy(isAIAbortError);
    });

    it('aborts a running operation by dedupeKey', async () => {
        const q = makeQueue();

        const p = q.enqueue('task', async (ctx) => {
            const d = signalAwareDeferred(ctx.signal);
            return d.promise;
        }, { dedupeKey: 'myKey' });
        await tick();

        q.cancelKey('myKey');
        await expect(p).rejects.toSatisfy(isAIAbortError);
    });
});

// ---------------------------------------------------------------------------
// hasPending
// ---------------------------------------------------------------------------

describe('hasPending', () => {
    it('returns true when operation is queued', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        q.enqueue('blocker', () => blocker.promise);
        await tick();

        q.enqueue('task', immediate(1), { dedupeKey: 'dk' });
        expect(q.hasPending('dk')).toBe(true);

        blocker.resolve();
        await flushQueue();
        expect(q.hasPending('dk')).toBe(false);
    });

    it('returns true when operation is running', async () => {
        const q = makeQueue();
        const d = deferred();
        q.enqueue('task', () => d.promise, { dedupeKey: 'dk' });
        await tick();

        expect(q.hasPending('dk')).toBe(true);
        d.resolve();
        await flushQueue();
        expect(q.hasPending('dk')).toBe(false);
    });

    it('returns false for unknown key', () => {
        const q = makeQueue();
        expect(q.hasPending('nope')).toBe(false);
    });
});

// ---------------------------------------------------------------------------
// setFocusedGroup
// ---------------------------------------------------------------------------

describe('setFocusedGroup', () => {
    it('sorts focused group ops to the front of the queue', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const order: string[] = [];

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        const pOther = q.enqueue('other', async () => { order.push('other'); }, {
            group: 'g-other',
            priority: 50,
        });
        const pFocused = q.enqueue('focused', async () => { order.push('focused'); }, {
            group: 'g-focused',
            priority: 10, // lower numeric priority but group is focused
        });

        q.setFocusedGroup('g-focused');

        blocker.resolve();
        await p0;
        await Promise.all([pOther, pFocused]);

        expect(order[0]).toBe('focused');
    });

    it('allows focused ops to use burst slots', async () => {
        // base=2, burst=1 → max 3 total, but non-urgent max 2
        const q = makeQueue({ base: 2, burst: 1 });
        const blockers = Array.from({ length: 2 }, () => deferred());

        // Fill base slots with non-urgent work
        blockers.forEach((b, i) =>
            q.enqueue(`bg${i}`, () => b.promise, { group: 'background', priority: 1 }),
        );
        await tick();
        expect(q.getStats().runningOperations).toBe(2);

        // Focus a group and enqueue urgent work — should use burst slot
        q.setFocusedGroup('fg');
        const focusedD = deferred();
        const pFocused = q.enqueue('focused', () => focusedD.promise, {
            group: 'fg',
            priority: 10,
        });

        await tick();
        // Burst slot should allow this to start despite base concurrency being full
        expect(q.getStats().runningOperations).toBe(3);

        focusedD.resolve();
        await pFocused;
        blockers.forEach((b) => b.resolve());
    });
});

// ---------------------------------------------------------------------------
// slowOp concurrency cap
// ---------------------------------------------------------------------------

describe('slowOp', () => {
    it('caps slow ops below the base concurrency', async () => {
        // base=3, slowOp=1 → at most 1 slow op in background
        const q = makeQueue({ base: 3, burst: 0, slowOp: 1 });
        const slow1 = deferred();
        const slow2 = deferred();

        q.enqueue('slow1', () => slow1.promise, { slowOp: true });
        await tick();
        expect(q.getStats().runningOperations).toBe(1);

        // Second slow op should queue (cap hit)
        q.enqueue('slow2', () => slow2.promise, { slowOp: true });
        await tick();
        expect(q.getStats().runningOperations).toBe(1);
        expect(q.getStats().pendingOperations).toBe(1);

        slow1.resolve();
        await flushQueue();
        // Now slow2 can start
        expect(q.getStats().runningOperations).toBe(1);

        slow2.resolve();
        await flushQueue();
    });

    it('does not cap fast ops even when slow ops are running', async () => {
        const q = makeQueue({ base: 3, burst: 0, slowOp: 1 });
        const slow = deferred();
        const fast1 = deferred();
        const fast2 = deferred();

        q.enqueue('slow', () => slow.promise, { slowOp: true });
        q.enqueue('fast1', () => fast1.promise);
        q.enqueue('fast2', () => fast2.promise);

        await tick();
        expect(q.getStats().runningOperations).toBe(3);

        slow.resolve();
        fast1.resolve();
        fast2.resolve();
        await flushQueue();
    });
});

// ---------------------------------------------------------------------------
// subscribe
// ---------------------------------------------------------------------------

describe('subscribe', () => {
    it('fires queued event when enqueuing', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const events: string[] = [];
        q.subscribe((e) => events.push(e.kind));

        const blocker = deferred();
        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        q.enqueue('task', immediate(1));
        blocker.resolve();
        await p0;
        await flushQueue();

        expect(events).toContain('queued');
    });

    it('fires started and completed events', async () => {
        const q = makeQueue();
        const events: string[] = [];
        q.subscribe((e) => events.push(e.kind));

        await q.enqueue('task', immediate(1));

        expect(events).toContain('started');
        expect(events).toContain('completed');
    });

    it('fires failed event on task error', async () => {
        const q = makeQueue();
        const events: string[] = [];
        q.subscribe((e) => events.push(e.kind));

        await q.enqueue('task', async () => { throw new Error('x'); }).catch(() => {});
        expect(events).toContain('failed');
    });

    it('fires skipped event when isStillValid is false', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const events: string[] = [];
        q.subscribe((e) => events.push(e.kind));

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();
        q.enqueue('task', immediate(1), { isStillValid: () => false });
        blocker.resolve();
        await p0;
        await flushQueue();

        expect(events).toContain('skipped');
    });

    it('fires cancelled event when cancelGroup is called', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const events: string[] = [];
        q.subscribe((e) => events.push(e.kind));

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();
        // This promise will be cancelled — catch the rejection so the test doesn't fail
        q.enqueue('task', immediate(1), { group: 'g' }).catch(() => {});
        q.cancelGroup('g');
        blocker.resolve();
        await p0;
        await flushQueue();

        expect(events).toContain('cancelled');
    });

    it('returns an unsubscribe function that stops events', async () => {
        const q = makeQueue();
        const events: string[] = [];
        const unsub = q.subscribe((e) => events.push(e.kind));

        await q.enqueue('t1', immediate(1));
        const countAfterFirst = events.length;

        unsub();
        await q.enqueue('t2', immediate(2));

        // No new events should have been pushed
        expect(events.length).toBe(countAfterFirst);
    });

    it('includes current stats in every event', async () => {
        const q = makeQueue();
        let lastStats = q.getStats();
        q.subscribe((e) => { lastStats = e.stats; });

        await q.enqueue('t', immediate(1));
        expect(lastStats.completedOperations).toBeGreaterThanOrEqual(1);
    });

    it('does not crash the queue if a listener throws', async () => {
        const q = makeQueue();
        q.subscribe(() => { throw new Error('listener error'); });

        // Queue should still work
        const result = await q.enqueue('t', immediate(42));
        expect(result).toBe(42);
    });
});

// ---------------------------------------------------------------------------
// drain
// ---------------------------------------------------------------------------

describe('drain()', () => {
    it('resolves immediately when queue is empty', async () => {
        const q = makeQueue();
        await expect(q.drain()).resolves.toBeUndefined();
    });

    it('resolves after all running operations finish', async () => {
        const q = makeQueue();
        const d1 = deferred();
        const d2 = deferred();
        let drained = false;

        q.enqueue('t1', () => d1.promise);
        q.enqueue('t2', () => d2.promise);
        await tick();

        const drainP = q.drain().then(() => { drained = true; });

        expect(drained).toBe(false);
        d1.resolve();
        d2.resolve();
        await drainP;
        expect(drained).toBe(true);
    });

    it('resolves after queued operations settle', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const d1 = deferred();
        const d2 = deferred();

        q.enqueue('t1', () => d1.promise);
        q.enqueue('t2', () => d2.promise);
        await tick();

        const drainP = q.drain();
        d1.resolve();
        await tick();
        d2.resolve();
        await drainP; // should resolve once both are done
    });
});

// ---------------------------------------------------------------------------
// destroy
// ---------------------------------------------------------------------------

describe('destroy()', () => {
    it('cancels all pending operations', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();

        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        const p1 = q.enqueue('pending', immediate(1));
        p1.catch(() => {});

        q.destroy('test-reason');
        blocker.resolve(); // blocker still completes since it was running before destroy

        await expect(p1).rejects.toSatisfy(isAIAbortError);
    });

    it('aborts running operations', async () => {
        const q = makeQueue();
        let aborted = false;

        const p = q.enqueue('running', async (ctx) => {
            const d = signalAwareDeferred(ctx.signal);
            ctx.signal.addEventListener('abort', () => { aborted = true; });
            return d.promise;
        });
        p.catch(() => {});

        await tick();
        q.destroy('shutdown');
        await tick();

        expect(aborted).toBe(true);
        await expect(p).rejects.toSatisfy(isAIAbortError);
    });

    it('makes subsequent enqueue() calls reject immediately', async () => {
        const q = makeQueue();
        q.destroy();
        await expect(q.enqueue('after-destroy', immediate(1))).rejects.toSatisfy(isAIAbortError);
    });

    it('is safe to call multiple times', () => {
        const q = makeQueue();
        expect(() => { q.destroy(); q.destroy(); q.destroy(); }).not.toThrow();
    });

    it('emits cancelled events for pending ops', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        const events: string[] = [];
        q.subscribe((e) => events.push(e.kind));

        q.enqueue('blocker', () => blocker.promise);
        await tick();

        q.enqueue('pending', immediate(1)).catch(() => {});
        q.destroy();
        blocker.resolve();
        await flushQueue();

        expect(events).toContain('cancelled');
    });
});

// ---------------------------------------------------------------------------
// runningLabels in stats
// ---------------------------------------------------------------------------

describe('getStats().runningLabels', () => {
    it('contains labels of all running operations', async () => {
        const q = makeQueue({ base: 2, burst: 0 });
        const d1 = deferred();
        const d2 = deferred();

        q.enqueue('task-alpha', () => d1.promise);
        q.enqueue('task-beta', () => d2.promise);
        await tick();

        const { runningLabels } = q.getStats();
        expect(runningLabels).toContain('task-alpha');
        expect(runningLabels).toContain('task-beta');

        d1.resolve();
        d2.resolve();
        await flushQueue();
    });

    it('is empty when nothing is running', () => {
        const q = makeQueue();
        expect(q.getStats().runningLabels).toEqual([]);
    });
});

// ---------------------------------------------------------------------------
// getStats
// ---------------------------------------------------------------------------

describe('getStats', () => {
    it('tracks all counters correctly', async () => {
        const q = makeQueue({ base: 1, burst: 0 });

        // 1 completed
        await q.enqueue('ok', immediate(1));
        // 1 failed
        await q.enqueue('fail', async () => { throw new Error('x'); }).catch(() => {});

        const blocker = deferred();
        const p0 = q.enqueue('blocker', () => blocker.promise);
        await tick();

        // 1 cancelled (queued)
        q.enqueue('cancelled', immediate(1), { group: 'g' }).catch(() => {});
        q.cancelGroup('g');

        // 1 skipped
        q.enqueue('skipped', immediate(1), { isStillValid: () => false });

        blocker.resolve();
        await p0;
        await flushQueue();

        const stats = q.getStats();
        expect(stats.completedOperations).toBe(2); // ok + blocker
        expect(stats.failedOperations).toBe(1);
        expect(stats.cancelledOperations).toBe(1);
        expect(stats.skippedOperations).toBe(1);
    });

    it('shows pendingByPriority breakdown', async () => {
        const q = makeQueue({ base: 1, burst: 0 });
        const blocker = deferred();
        q.enqueue('blocker', () => blocker.promise);
        await tick();

        q.enqueue('low', immediate(1), { priority: 10 });
        q.enqueue('high', immediate(1), { priority: 50 });
        q.enqueue('high2', immediate(1), { priority: 50 });

        const stats = q.getStats();
        expect(stats.pendingByPriority['10']).toBe(1);
        expect(stats.pendingByPriority['50']).toBe(2);

        blocker.resolve();
        await flushQueue(10);
    });
});

// ---------------------------------------------------------------------------
// recordUsage / usage in stats
// ---------------------------------------------------------------------------

describe('recordUsage', () => {
    it('aggregates token and cost totals', () => {
        const q = makeQueue();
        q.recordUsage(100, 50, 0.001);
        q.recordUsage(200, 100, 0.002);

        const { usage } = q.getStats();
        expect(usage.promptTokens).toBe(300);
        expect(usage.completionTokens).toBe(150);
        expect(usage.totalTokens).toBe(450);
        expect(usage.estimatedCostUsd).toBeCloseTo(0.003);
    });

    it('handles undefined cost gracefully', () => {
        const q = makeQueue();
        q.recordUsage(100, 50, undefined);
        expect(q.getStats().usage.estimatedCostUsd).toBe(0);
    });
});

// ---------------------------------------------------------------------------
// AbortSignal propagation
// ---------------------------------------------------------------------------

describe('AbortSignal propagation', () => {
    it('signal is aborted when cancelGroup is called mid-task', async () => {
        const q = makeQueue();
        const abortedValues: unknown[] = [];

        const p = q.enqueue('task', async (ctx) => {
            const d = signalAwareDeferred(ctx.signal);
            ctx.signal.addEventListener('abort', () => {
                abortedValues.push(ctx.signal.reason);
            });
            return d.promise;
        }, { group: 'g' });
        p.catch(() => {});

        await tick();
        q.cancelGroup('g', 'test reason');
        await tick();

        expect(abortedValues).toHaveLength(1);
        expect(abortedValues[0]).toBeInstanceOf(AIOperationCancelledError);
        await expect(p).rejects.toSatisfy(isAIAbortError);
    });
});
