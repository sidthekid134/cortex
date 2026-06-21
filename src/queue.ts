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

export class AIOperationCancelledError extends Error {
    constructor(message = 'AI operation cancelled') {
        super(message);
        this.name = 'AIOperationCancelledError';
    }
}

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
    onEvent?: (event: QueueEvent) => void;
}

// ---------------------------------------------------------------------------
// Usage totals (written into QueueStats)
// ---------------------------------------------------------------------------

export interface UsageTotals {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    estimatedCostUsd: number;
}

// ---------------------------------------------------------------------------
// AIOperationQueue
// ---------------------------------------------------------------------------

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

    private readonly cfg: ResolvedQueueConfig;

    constructor(concurrency?: ConcurrencyConfig, interactiveThreshold?: number, logger?: Logger) {
        const base = concurrency?.base ?? 3;
        const burst = concurrency?.burst ?? 2;
        this.cfg = {
            baseConcurrency: base,
            burstSlots: burst,
            slowOpConcurrency: concurrency?.slowOp ?? base - 1,
            interactiveThreshold: interactiveThreshold ?? 90,
            logger: logger ?? noopLogger,
        };
    }

    // -------------------------------------------------------------------------
    // Public: subscribe
    // -------------------------------------------------------------------------

    /**
     * Subscribe to queue lifecycle events (queued, started, completed, etc.).
     * Returns an unsubscribe function. Apps use this to mirror queue state into
     * their own stores without polling getStats().
     */
    subscribe(listener: QueueListener): () => void {
        this.listeners.add(listener);
        return () => { this.listeners.delete(listener); };
    }

    // -------------------------------------------------------------------------
    // Public: enqueue
    // -------------------------------------------------------------------------

    enqueue<T>(
        label: string,
        task: (ctx: AITaskContext) => Promise<T>,
        options: AIOperationOptions = {},
    ): Promise<T> {
        const dedupeKey = options.dedupeKey ?? label;
        const priority = options.priority ?? 0;

        // If the same key is already running, attach to it or restart it.
        const runningMatch = [...this.running.values()].find((e) => e.op.dedupeKey === dedupeKey);
        if (runningMatch) {
            if (options.restart) {
                this.abortRunning(runningMatch.op.id, 'restart');
            } else {
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

        // If the same key is pending in the queue, bump it and replace the task.
        const existing = this.queue.find((op) => op.dedupeKey === dedupeKey);
        if (existing) {
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
            void this.processNext();
        });
    }

    // -------------------------------------------------------------------------
    // Public: focus / cancel / query
    // -------------------------------------------------------------------------

    /**
     * Mark the group the user is actively viewing. Its queued operations are
     * sorted to the front and may run in burst slots without cancelling
     * already-running work.
     */
    setFocusedGroup(group: string | null): void {
        if (this.focusedGroup === group) return;
        this.focusedGroup = group;
        this.sortQueue();
        void this.processNext();
    }

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
            this.cfg.logger.trace(`[cortex] cancelled group ${group}: ${cancelled} op(s) — ${reason}`);
        }
    }

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
            this.cfg.logger.trace(`[cortex] cancelled key "${dedupeKey}": ${cancelled} op(s) — ${reason}`);
        }
    }

    hasPending(dedupeKey: string): boolean {
        if (!dedupeKey) return false;
        for (const entry of this.running.values()) {
            if (entry.op.dedupeKey === dedupeKey) return true;
        }
        return this.queue.some((op) => op.dedupeKey === dedupeKey);
    }

    getStats(): QueueStats {
        const first = this.running.values().next().value as RunningOperation | undefined;
        return {
            activeOperation: first?.op.label ?? null,
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

    /** Called by telemetry after a completed LLM call to accumulate totals. */
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
                continue;
            }

            void this.runOperation(next);
        }
    }

    private async runOperation(op: QueuedOperation<unknown>): Promise<void> {
        const controller = new AbortController();
        this.running.set(op.id, { op, controller });
        this.emit('started', op);
        this.cfg.logger.trace(`[cortex] ▶ started  "${op.label}" p${op.priority} — running ${this.running.size}, waiting ${this.queue.length}`);

        try {
            await yieldToUI();
            if (controller.signal.aborted) {
                throw controller.signal.reason instanceof Error
                    ? controller.signal.reason
                    : new AIOperationCancelledError(op.label);
            }

            const result = await op.task({ signal: controller.signal });
            this.completedOperations++;
            op.resolve(result);
            this.emit('completed', op);
            this.cfg.logger.trace(`[cortex] ✓ complete "${op.label}"`);
        } catch (error) {
            if (isAIAbortError(error) || controller.signal.aborted) {
                op.reject(error instanceof Error ? error : new AIOperationCancelledError(op.label));
                this.emit('cancelled', op);
                this.cfg.logger.debug(`[cortex] ✕ cancelled "${op.label}"`);
            } else {
                this.failedOperations++;
                op.reject(error);
                this.emit('failed', op);
                this.cfg.logger.error(`[cortex] ✕ failed   "${op.label}"`, error);
            }
        } finally {
            this.running.delete(op.id);
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

    private emit(kind: QueueEvent['kind'], info: { label?: string; group?: string; priority?: number }): void {
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
