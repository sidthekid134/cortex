import type {
    Logger,
    AITaskContext,
    AIOperationOptions,
    QueueStats,
    QueueEvent,
    QueueListener,
    ConcurrencyConfig,
} from './types.js';
import { noopLogger } from './logger.js';

const yieldToUI = (): Promise<void> => new Promise<void>((resolve) => setTimeout(resolve, 0));

// ---------------------------------------------------------------------------
// Cancellation error
// ---------------------------------------------------------------------------

/**
 * Thrown when an operation is cancelled via `cancelGroup()`, `cancelKey()`,
 * or `destroy()`. Also used as the abort reason on `ctx.signal`.
 */
export class AIOperationCancelledError extends Error {
    constructor(message = 'AI operation cancelled') {
        super(message);
        this.name = 'AIOperationCancelledError';
    }
}

/**
 * Returns `true` for any cancellation-originated error, whether it came from
 * `AIOperationCancelledError`, a browser `AbortError`, or a signal abort.
 * Use this to distinguish intentional cancellations from real failures.
 */
export const isAIAbortError = (error: unknown): boolean => {
    if (error instanceof AIOperationCancelledError) return true;
    if (error instanceof Error) {
        return (
            error.name === 'AbortError' ||
            error.name === 'AIOperationCancelledError' ||
            /\baborted\b/i.test(error.message)
        );
    }
    return false;
};

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

interface QueuedOperation<T> {
    id: number;
    label: string;
    dedupeKey: string;
    group?: string;
    priority: number;
    createdAt: number;
    task: (ctx: AITaskContext) => Promise<T>;
    isStillValid?: () => boolean;
    slowOp?: boolean;
    resolve: (value: T | PromiseLike<T>) => void;
    reject: (reason?: unknown) => void;
}

interface RunningOperation {
    op: QueuedOperation<unknown>;
    controller: AbortController;
}

// ---------------------------------------------------------------------------
// Queue config (resolved from ConcurrencyConfig + interactiveThreshold)
// ---------------------------------------------------------------------------

interface ResolvedQueueConfig {
    baseConcurrency: number;
    burstSlots: number;
    slowOpConcurrency: number;
    interactiveThreshold: number;
    logger: Logger;
}

// ---------------------------------------------------------------------------
// Usage totals (internal — aggregated into QueueStats)
// ---------------------------------------------------------------------------

interface UsageTotals {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    estimatedCostUsd: number;
}

// ---------------------------------------------------------------------------
// AIOperationQueue
// ---------------------------------------------------------------------------

/**
 * Priority-aware, deduplication-supporting operation queue for async LLM tasks.
 *
 * Operations are scheduled by priority and focused-group status. Slow operations
 * are limited to a separate concurrency cap so fast operations are never starved.
 * Cancellation propagates through `AbortSignal` on each task's context.
 *
 * Prefer using the `CortexInstance` returned by `createAIQueue()` rather than
 * instantiating this class directly.
 */
export class AIOperationQueue {
    private queue: QueuedOperation<unknown>[] = [];
    private running = new Map<number, RunningOperation>();
    private nextId = 1;
    private completedOperations = 0;
    private failedOperations = 0;
    private skippedOperations = 0;
    private cancelledOperations = 0;
    private usageTotals: UsageTotals = {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        estimatedCostUsd: 0,
    };
    private focusedGroup: string | null = null;
    private listeners: Set<QueueListener> = new Set();
    private destroyed = false;

    private readonly cfg: ResolvedQueueConfig;

    constructor(concurrency?: ConcurrencyConfig, interactiveThreshold?: number, logger?: Logger) {
        const base = concurrency?.base ?? 3;
        const burst = concurrency?.burst ?? 2;
        this.cfg = {
            baseConcurrency: base,
            burstSlots: burst,
            slowOpConcurrency: concurrency?.slowOp ?? Math.max(1, base - 1),
            interactiveThreshold: interactiveThreshold ?? 90,
            logger: logger ?? noopLogger,
        };
    }

    // -------------------------------------------------------------------------
    // Public: subscribe
    // -------------------------------------------------------------------------

    /**
     * Subscribe to queue lifecycle events (`queued`, `started`, `completed`, etc.).
     * Returns an unsubscribe function. Use this to mirror queue state into your
     * own store without polling `getStats()`.
     *
     * Listener errors are swallowed — a buggy listener will never crash the queue.
     */
    subscribe(listener: QueueListener): () => void {
        this.listeners.add(listener);
        return () => { this.listeners.delete(listener); };
    }

    // -------------------------------------------------------------------------
    // Public: enqueue
    // -------------------------------------------------------------------------

    /**
     * Add a task to the queue. The task receives an `AITaskContext` with a
     * cancellation signal and should pass `ctx.signal` to any in-flight I/O.
     *
     * If an operation with the same `dedupeKey` is already pending, the new
     * call supersedes it: the task function and priority are updated, and both
     * promises resolve/reject together.
     *
     * @throws {AIOperationCancelledError} if the queue has been destroyed.
     */
    enqueue<T>(
        label: string,
        task: (ctx: AITaskContext) => Promise<T>,
        options: AIOperationOptions = {},
    ): Promise<T> {
        if (this.destroyed) {
            return Promise.reject(
                new AIOperationCancelledError('[cortex] Queue has been destroyed'),
            );
        }

        const dedupeKey = options.dedupeKey ?? label;
        const priority = options.priority ?? 0;

        // Already running — attach to the running op or restart it.
        const runningMatch = [...this.running.values()].find((e) => e.op.dedupeKey === dedupeKey);
        if (runningMatch) {
            if (options.restart) {
                this.cfg.logger.debug(`enqueue "${label}" — restarting running op`);
                this.abortRunning(runningMatch.op.id, 'restart');
            } else {
                this.cfg.logger.trace(`enqueue "${label}" — attaching to running op`);
                const runningOp = runningMatch.op;
                runningOp.priority = Math.max(runningOp.priority, priority);
                runningOp.label = label;
                return new Promise<T>((resolve, reject) => {
                    const prevResolve = runningOp.resolve;
                    const prevReject = runningOp.reject;
                    runningOp.resolve = (value) => { prevResolve(value); resolve(value as T); };
                    runningOp.reject = (reason) => { prevReject(reason); reject(reason); };
                });
            }
        }

        // Already pending — bump priority and update the task.
        const existing = this.queue.find((op) => op.dedupeKey === dedupeKey);
        if (existing) {
            this.cfg.logger.trace(`enqueue "${label}" — bumping pending op`);
            existing.priority = Math.max(existing.priority, priority);
            existing.label = label;
            existing.task = task as (ctx: AITaskContext) => Promise<unknown>;
            existing.isStillValid = options.isStillValid;
            existing.group = options.group ?? existing.group;
            this.sortQueue();
            this.emit('bumped', existing);
            void this.processNext();
            return new Promise<T>((resolve, reject) => {
                const prevResolve = existing.resolve;
                const prevReject = existing.reject;
                existing.resolve = (value) => { prevResolve(value); resolve(value as T); };
                existing.reject = (reason) => { prevReject(reason); reject(reason); };
            });
        }

        // New operation.
        return new Promise<T>((resolve, reject) => {
            this.queue.push({
                id: this.nextId++,
                label,
                dedupeKey,
                group: options.group,
                priority,
                createdAt: Date.now(),
                task: task as (ctx: AITaskContext) => Promise<unknown>,
                isStillValid: options.isStillValid,
                slowOp: options.slowOp,
                resolve: resolve as (value: unknown) => void,
                reject,
            });
            this.sortQueue();
            this.emit('queued', { label, group: options.group, priority });
            this.cfg.logger.trace(
                `enqueue "${label}" p${priority} — queue ${this.queue.length}, running ${this.running.size}`,
            );
            void this.processNext();
        });
    }

    // -------------------------------------------------------------------------
    // Public: focus / cancel / query
    // -------------------------------------------------------------------------

    /**
     * Mark the group the user is actively viewing. Its queued operations are
     * sorted to the front and are allowed to use burst concurrency slots without
     * aborting already-running work.
     */
    setFocusedGroup(group: string | null): void {
        if (this.focusedGroup === group) return;
        this.focusedGroup = group;
        this.cfg.logger.debug(`focusedGroup → ${group ?? 'null'}`);
        this.sortQueue();
        void this.processNext();
    }

    /**
     * Cancel all queued **and** running operations belonging to `group`.
     * The focused group is cleared if it matches.
     */
    cancelGroup(group: string, reason = 'group cancelled'): void {
        if (!group) return;
        if (this.focusedGroup === group) this.focusedGroup = null;

        let cancelled = 0;
        const remaining: QueuedOperation<unknown>[] = [];
        for (const op of this.queue) {
            if (op.group === group) {
                this.cancelledOperations++;
                cancelled++;
                op.reject(new AIOperationCancelledError(`${op.label}: ${reason}`));
                this.emit('cancelled', op);
            } else {
                remaining.push(op);
            }
        }
        this.queue = remaining;

        for (const entry of [...this.running.values()]) {
            if (entry.op.group === group) {
                cancelled++;
                this.abortRunning(entry.op.id, reason);
            }
        }

        if (cancelled > 0) {
            this.cfg.logger.debug(`cancelGroup "${group}" — ${cancelled} op(s): ${reason}`);
        }
    }

    /**
     * Cancel all queued **and** running operations matching a dedupe key.
     */
    cancelKey(dedupeKey: string, reason = 'key cancelled'): void {
        if (!dedupeKey) return;

        let cancelled = 0;
        const remaining: QueuedOperation<unknown>[] = [];
        for (const op of this.queue) {
            if (op.dedupeKey === dedupeKey) {
                this.cancelledOperations++;
                cancelled++;
                op.reject(new AIOperationCancelledError(`${op.label}: ${reason}`));
                this.emit('cancelled', op);
            } else {
                remaining.push(op);
            }
        }
        this.queue = remaining;

        for (const entry of [...this.running.values()]) {
            if (entry.op.dedupeKey === dedupeKey) {
                cancelled++;
                this.abortRunning(entry.op.id, reason);
            }
        }

        if (cancelled > 0) {
            this.cfg.logger.debug(`cancelKey "${dedupeKey}" — ${cancelled} op(s): ${reason}`);
        }
    }

    /**
     * Returns `true` if any operation with the given dedupe key is currently
     * queued or running.
     */
    hasPending(dedupeKey: string): boolean {
        if (!dedupeKey) return false;
        for (const entry of this.running.values()) {
            if (entry.op.dedupeKey === dedupeKey) return true;
        }
        return this.queue.some((op) => op.dedupeKey === dedupeKey);
    }

    /** Snapshot of the current queue state, including aggregate usage totals. */
    getStats(): QueueStats {
        const runningList = [...this.running.values()];
        return {
            activeOperation: runningList[0]?.op.label ?? null,
            runningLabels: runningList.map((e) => e.op.label),
            runningOperations: this.running.size,
            pendingOperations: this.queue.length,
            pendingByPriority: this.queue.reduce<Record<string, number>>((acc, op) => {
                const k = String(op.priority);
                acc[k] = (acc[k] ?? 0) + 1;
                return acc;
            }, {}),
            completedOperations: this.completedOperations,
            failedOperations: this.failedOperations,
            skippedOperations: this.skippedOperations,
            cancelledOperations: this.cancelledOperations,
            usage: { ...this.usageTotals },
        };
    }

    /**
     * Wait for all currently running and queued operations to settle (resolve or
     * reject). Resolves immediately if the queue is already empty. Useful for
     * testing and graceful shutdown.
     */
    drain(): Promise<void> {
        if (this.running.size === 0 && this.queue.length === 0) {
            return Promise.resolve();
        }
        return new Promise<void>((resolve) => {
            const unsubscribe = this.subscribe(() => {
                if (this.running.size === 0 && this.queue.length === 0) {
                    unsubscribe();
                    resolve();
                }
            });
        });
    }

    /**
     * Cancel all pending and running operations, then mark the queue as
     * destroyed. After calling `destroy()`, `enqueue()` will throw immediately.
     *
     * Safe to call multiple times — subsequent calls are no-ops.
     */
    destroy(reason = 'queue destroyed'): void {
        if (this.destroyed) return;
        this.destroyed = true;

        this.cfg.logger.info(`destroy — cancelling all operations: ${reason}`);

        for (const op of this.queue) {
            this.cancelledOperations++;
            op.reject(new AIOperationCancelledError(`${op.label}: ${reason}`));
            this.emit('cancelled', op);
        }
        this.queue = [];

        for (const entry of this.running.values()) {
            this.cancelledOperations++;
            entry.controller.abort(
                new AIOperationCancelledError(`${entry.op.label}: ${reason}`),
            );
        }
    }

    /** Called by telemetry after a completed LLM call to accumulate session totals. */
    recordUsage(promptTokens: number, completionTokens: number, costUsd: number | undefined): void {
        this.usageTotals.promptTokens += promptTokens;
        this.usageTotals.completionTokens += completionTokens;
        this.usageTotals.totalTokens += promptTokens + completionTokens;
        this.usageTotals.estimatedCostUsd += costUsd ?? 0;
    }

    // -------------------------------------------------------------------------
    // Private
    // -------------------------------------------------------------------------

    private isUrgent(op: QueuedOperation<unknown>): boolean {
        return (
            (!!op.group && op.group === this.focusedGroup) ||
            op.priority >= this.cfg.interactiveThreshold
        );
    }

    private canStart(op: QueuedOperation<unknown>): boolean {
        const max = this.cfg.baseConcurrency + this.cfg.burstSlots;
        if (this.running.size >= max) return false;
        if (this.isUrgent(op)) return true;
        const bgRunning = [...this.running.values()].filter((e) => !this.isUrgent(e.op)).length;
        if (bgRunning >= this.cfg.baseConcurrency) return false;
        if (op.slowOp) {
            const slowRunning = [...this.running.values()].filter(
                (e) => e.op.slowOp && !this.isUrgent(e.op),
            ).length;
            if (slowRunning >= this.cfg.slowOpConcurrency) return false;
        }
        return true;
    }

    private async processNext(): Promise<void> {
        while (this.queue.length > 0) {
            const next = this.queue[0];
            if (!next || !this.canStart(next)) break;
            this.queue.shift();

            if (next.isStillValid && !next.isStillValid()) {
                this.skippedOperations++;
                next.resolve(undefined);
                this.emit('skipped', next);
                this.cfg.logger.trace(`skipped "${next.label}" (isStillValid → false)`);
                continue;
            }

            void this.runOperation(next);
        }
    }

    private async runOperation(op: QueuedOperation<unknown>): Promise<void> {
        const controller = new AbortController();
        this.running.set(op.id, { op, controller });
        this.emit('started', op);
        this.cfg.logger.trace(
            `▶ "${op.label}" p${op.priority} — running ${this.running.size}, waiting ${this.queue.length}`,
        );

        try {
            await yieldToUI();
            if (controller.signal.aborted) {
                throw controller.signal.reason instanceof Error
                    ? controller.signal.reason
                    : new AIOperationCancelledError(op.label);
            }

            const result = await op.task({ signal: controller.signal });
            this.completedOperations++;
            // Delete before emitting so drain() sees the correct running count.
            this.running.delete(op.id);
            op.resolve(result);
            this.emit('completed', op);
            this.cfg.logger.trace(`✓ "${op.label}"`);
        } catch (error) {
            // Delete before emitting so drain() sees the correct running count.
            this.running.delete(op.id);
            if (isAIAbortError(error) || controller.signal.aborted) {
                op.reject(error instanceof Error ? error : new AIOperationCancelledError(op.label));
                this.emit('cancelled', op);
                this.cfg.logger.debug(`✕ cancelled "${op.label}"`);
            } else {
                this.failedOperations++;
                op.reject(error);
                this.emit('failed', op);
                this.cfg.logger.error(`✕ failed "${op.label}"`, error);
            }
        } finally {
            // Guard: already deleted in the try/catch above, but keep processNext here.
            this.running.delete(op.id); // no-op if already deleted; safe to call twice
            await yieldToUI();
            void this.processNext();
        }
    }

    private sortQueue(): void {
        const focused = this.focusedGroup;
        this.queue.sort((a, b) => {
            if (focused) {
                const aF = a.group === focused ? 1 : 0;
                const bF = b.group === focused ? 1 : 0;
                if (aF !== bF) return bF - aF;
            }
            if (b.priority !== a.priority) return b.priority - a.priority;
            return a.createdAt - b.createdAt;
        });
    }

    private abortRunning(id: number, reason: string): void {
        const entry = this.running.get(id);
        if (!entry) return;
        this.cancelledOperations++;
        entry.controller.abort(new AIOperationCancelledError(`${entry.op.label}: ${reason}`));
    }

    private emit(
        kind: QueueEvent['kind'],
        info: { label?: string; group?: string; priority?: number },
    ): void {
        if (this.listeners.size === 0) return;
        const event: QueueEvent = {
            kind,
            label: info.label ?? '',
            group: info.group,
            priority: info.priority,
            stats: this.getStats(),
        };
        for (const listener of this.listeners) {
            try { listener(event); } catch { /* never let a listener crash the queue */ }
        }
    }
}
