import type { CortexConfig, StructuredParams, TextParams, UsageEvent } from './types.js';
import type { ZodTypeAny } from 'zod';
import type { z } from 'zod';
import { AIOperationQueue, isAIAbortError, AIOperationCancelledError } from './queue.js';
import { OpenRouterClient } from './openrouter.js';
import { resolvePresets } from './presets.js';
import { structured, text } from './structured.js';
import type { AIOperationOptions, QueueStats, QueueListener } from './types.js';
import type { AITaskContext } from './types.js';

export interface CortexInstance {
    // ------------------------------------------------------------------
    // Queue management (full surface for app-controlled orchestration)
    // ------------------------------------------------------------------

    /**
     * Add a task to the queue. The task receives an AbortSignal via ctx and
     * should pass it through to any LLM calls (ai.structured / ai.text).
     */
    enqueue<T>(
        label: string,
        task: (ctx: AITaskContext) => Promise<T>,
        options?: AIOperationOptions,
    ): Promise<T>;

    /**
     * Mark the group the user is actively viewing. Its queued operations are
     * sorted to the front and may run in burst slots.
     */
    setFocusedGroup(group: string | null): void;

    /** Cancel all queued and running operations belonging to a group. */
    cancelGroup(group: string, reason?: string): void;

    /** Cancel queued and running operations matching a dedupe key. */
    cancelKey(dedupeKey: string, reason?: string): void;

    /** Returns true if any operation with this dedupe key is queued or running. */
    hasPending(dedupeKey: string): boolean;

    /** Current queue statistics including aggregate usage totals. */
    getStats(): QueueStats;

    /**
     * Subscribe to queue lifecycle events (queued, started, completed, etc.).
     * Returns an unsubscribe function. Use this to mirror queue state into
     * your own store without polling.
     */
    subscribe(listener: QueueListener): () => void;

    // ------------------------------------------------------------------
    // LLM call helpers
    // ------------------------------------------------------------------

    /**
     * Make a structured LLM call using a Zod schema. The model is asked for
     * JSON output matching the schema; the response is validated and retried
     * once on mismatch. Returns the typed result.
     *
     * Designed to be called inside a queued task, with ctx.signal passed
     * through so cancellation propagates to the in-flight request.
     */
    structured<T extends ZodTypeAny>(params: StructuredParams<T>): Promise<z.infer<T>>;

    /**
     * Make a plain-text LLM call. No schema validation — the app handles
     * the response string directly.
     */
    text(params: TextParams): Promise<string>;
}

/**
 * Create a cortex instance. Call once at app startup (e.g. alongside your
 * store initialization) and share the instance via context or a module singleton.
 *
 * @example
 * const ai = createAIQueue({
 *   apiKey: OPENROUTER_API_KEY,
 *   headers: { referer: 'https://myapp.com', title: 'MyApp' },
 *   onUsage: (e) => console.log(`${e.preset} cost $${e.costUsd}`),
 * });
 */
export function createAIQueue(config: CortexConfig): CortexInstance {
    const presets = resolvePresets(config.presets);

    const queue = new AIOperationQueue(
        config.concurrency,
        config.interactiveThreshold,
        config.logger,
    );

    const client = new OpenRouterClient({
        apiKey: config.apiKey,
        baseUrl: config.baseUrl ?? 'https://openrouter.ai/api/v1',
        referer: config.headers?.referer,
        title: config.headers?.title,
    });

    const onUsage: ((e: UsageEvent) => void) | undefined = config.onUsage;

    return {
        enqueue: (label, task, options) => queue.enqueue(label, task, options),
        setFocusedGroup: (group) => queue.setFocusedGroup(group),
        cancelGroup: (group, reason) => queue.cancelGroup(group, reason),
        cancelKey: (dedupeKey, reason) => queue.cancelKey(dedupeKey, reason),
        hasPending: (dedupeKey) => queue.hasPending(dedupeKey),
        getStats: () => queue.getStats(),
        subscribe: (listener) => queue.subscribe(listener),

        structured: (params) => structured(params, presets, client, queue, onUsage),
        text: (params) => text(params, presets, client, queue, onUsage),
    };
}
