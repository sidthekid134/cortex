import type { CortexConfig, StructuredParams, TextParams, UsageEvent } from './types.js';
import type { ZodTypeAny } from 'zod';
import type { z } from 'zod';
import { AIOperationQueue, isAIAbortError, AIOperationCancelledError } from './queue.js';
import { OpenRouterClient } from './openrouter.js';
import { resolvePresets } from './presets.js';
import { structured, text } from './structured.js';
import type { AIOperationOptions, QueueStats, QueueListener } from './types.js';
import type { AITaskContext } from './types.js';

// ---------------------------------------------------------------------------
// CortexInstance
// ---------------------------------------------------------------------------

/**
 * The object returned by `createAIQueue()`. Exposes the full queue management
 * surface and LLM call helpers. Share a single instance across your app via
 * context or a module-level singleton.
 */
export interface CortexInstance {
    // ------------------------------------------------------------------
    // Queue management
    // ------------------------------------------------------------------

    /**
     * Add a task to the queue. The task function receives `ctx.signal` which is
     * aborted if the operation is cancelled. Pass `ctx.signal` through to
     * `ai.structured()`, `ai.text()`, and any other async I/O.
     *
     * @example
     * const result = await ai.enqueue('recognize-meal', async (ctx) => {
     *   return ai.structured({ preset: 'fast-vision', schema, prompt, signal: ctx.signal });
     * }, { group: mealId, dedupeKey: `recognize-${mealId}` });
     */
    enqueue<T>(
        label: string,
        task: (ctx: AITaskContext) => Promise<T>,
        options?: AIOperationOptions,
    ): Promise<T>;

    /**
     * Mark the group the user is actively viewing. Its queued operations are
     * sorted to the front and may run in burst concurrency slots.
     */
    setFocusedGroup(group: string | null): void;

    /** Cancel all queued and running operations belonging to `group`. */
    cancelGroup(group: string, reason?: string): void;

    /** Cancel queued and running operations matching a dedupe key. */
    cancelKey(dedupeKey: string, reason?: string): void;

    /**
     * Returns `true` if any operation with the given dedupe key is currently
     * queued or running.
     */
    hasPending(dedupeKey: string): boolean;

    /** Current queue statistics, including aggregate usage totals for this session. */
    getStats(): QueueStats;

    /**
     * Subscribe to queue lifecycle events (`queued`, `started`, `completed`, etc.).
     * Returns an unsubscribe function. Use this to mirror queue state into your
     * own store without polling.
     *
     * @example
     * const unsub = ai.subscribe((event) => {
     *   if (event.kind === 'completed') updateStore(event.stats);
     * });
     * // Later:
     * unsub();
     */
    subscribe(listener: QueueListener): () => void;

    /**
     * Wait for all currently running and queued operations to settle.
     * Useful for graceful shutdown and testing.
     */
    drain(): Promise<void>;

    /**
     * Cancel all pending and running operations, then mark the queue as
     * destroyed. Call this when the app no longer needs the queue
     * (e.g. on user logout or unmount). After `destroy()`, `enqueue()` throws.
     *
     * Safe to call multiple times.
     */
    destroy(reason?: string): void;

    // ------------------------------------------------------------------
    // LLM call helpers
    // ------------------------------------------------------------------

    /**
     * Make a structured LLM call using a Zod schema. The model is instructed to
     * respond with JSON matching the schema; the response is validated and
     * retried once on mismatch.
     *
     * Designed to be called inside a queued task with `ctx.signal` forwarded:
     * ```ts
     * ai.enqueue('label', async (ctx) =>
     *   ai.structured({ preset: 'fast-vision', schema, prompt, signal: ctx.signal })
     * );
     * ```
     *
     * @throws {StructuredOutputError} when all retries are exhausted.
     * @throws {AIOperationCancelledError} when the signal is aborted.
     * @throws {OpenRouterError} on non-retryable HTTP errors.
     */
    structured<T extends ZodTypeAny>(params: StructuredParams<T>): Promise<z.infer<T>>;

    /**
     * Make a plain-text LLM call with no schema validation.
     *
     * @throws {AIOperationCancelledError} when the signal is aborted.
     * @throws {OpenRouterError} on HTTP errors.
     */
    text(params: TextParams): Promise<string>;
}

// ---------------------------------------------------------------------------
// createAIQueue
// ---------------------------------------------------------------------------

/**
 * Create a cortex instance. Call once at app startup and share the instance
 * via context or a module-level singleton.
 *
 * @throws {Error} if `apiKey` is missing or concurrency values are invalid.
 *
 * @example
 * const ai = createAIQueue({
 *   apiKey: process.env.OPENROUTER_API_KEY!,
 *   headers: { referer: 'https://myapp.com', title: 'MyApp' },
 *   logger: createLogger('warn'),
 *   onUsage: (e) => analytics.track('llm_call', e),
 * });
 */
export function createAIQueue(config: CortexConfig): CortexInstance {
    // ------------------------------------------------------------------
    // Config validation
    // ------------------------------------------------------------------

    if (!config.apiKey || typeof config.apiKey !== 'string') {
        throw new Error('[cortex] createAIQueue: apiKey is required and must be a non-empty string');
    }

    const c = config.concurrency;
    if (c?.base !== undefined && (!Number.isInteger(c.base) || c.base < 1)) {
        throw new Error('[cortex] createAIQueue: concurrency.base must be a positive integer');
    }
    if (c?.burst !== undefined && (!Number.isInteger(c.burst) || c.burst < 0)) {
        throw new Error('[cortex] createAIQueue: concurrency.burst must be a non-negative integer');
    }
    if (c?.slowOp !== undefined && (!Number.isInteger(c.slowOp) || c.slowOp < 1)) {
        throw new Error('[cortex] createAIQueue: concurrency.slowOp must be a positive integer');
    }
    if (
        config.interactiveThreshold !== undefined &&
        (!Number.isInteger(config.interactiveThreshold) || config.interactiveThreshold < 0)
    ) {
        throw new Error('[cortex] createAIQueue: interactiveThreshold must be a non-negative integer');
    }

    // ------------------------------------------------------------------
    // Wire up internals
    // ------------------------------------------------------------------

    const logger = config.logger;
    const presets = resolvePresets(config.presets);

    const queue = new AIOperationQueue(
        config.concurrency,
        config.interactiveThreshold,
        logger,
    );

    const client = new OpenRouterClient({
        apiKey: config.apiKey,
        baseUrl: config.baseUrl ?? 'https://openrouter.ai/api/v1',
        referer: config.headers?.referer,
        title: config.headers?.title,
        logger,
    });

    const onUsage: ((e: UsageEvent) => void) | undefined = config.onUsage;

    return {
        // Queue
        enqueue: (label, task, options) => queue.enqueue(label, task, options),
        setFocusedGroup: (group) => queue.setFocusedGroup(group),
        cancelGroup: (group, reason) => queue.cancelGroup(group, reason),
        cancelKey: (dedupeKey, reason) => queue.cancelKey(dedupeKey, reason),
        hasPending: (dedupeKey) => queue.hasPending(dedupeKey),
        getStats: () => queue.getStats(),
        subscribe: (listener) => queue.subscribe(listener),
        drain: () => queue.drain(),
        destroy: (reason) => queue.destroy(reason),

        // LLM helpers
        structured: (params) => structured(params, presets, client, queue, onUsage, logger),
        text: (params) => text(params, presets, client, queue, onUsage, logger),
    };
}

// Re-export for convenience so callers don't have to import from two places.
export { isAIAbortError, AIOperationCancelledError };
