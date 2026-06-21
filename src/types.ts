import type { ZodTypeAny } from 'zod';

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

/** Verbosity level, from most verbose (`trace`) to silent (`silent`). */
export type LogLevel = 'trace' | 'debug' | 'info' | 'warn' | 'error' | 'silent';

/**
 * Minimal logging interface. Any logger that satisfies this shape (pino, winston,
 * console, etc.) can be passed to `createAIQueue`.
 */
export interface Logger {
    trace(message: string, ...args: unknown[]): void;
    debug(message: string, ...args: unknown[]): void;
    info(message: string, ...args: unknown[]): void;
    warn(message: string, ...args: unknown[]): void;
    error(message: string, ...args: unknown[]): void;
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

/**
 * A named model configuration. Apps choose a preset by name per operation;
 * the library resolves the actual model and parameters to use at call time.
 */
export interface Preset {
    /**
     * Primary model identifier (OpenRouter format: `"provider/model-name"`).
     * @example "google/gemini-2.0-flash"
     */
    model: string;
    /**
     * Optional fallback chain. OpenRouter will try each model in order when the
     * primary fails. Sent as the `models` array in the request body.
     */
    fallbacks?: string[];
    /** Sampling temperature (0–2). Lower values are more deterministic. */
    temperature?: number;
    /** Maximum tokens the model may generate in its response. */
    maxTokens?: number;
}

/** Map of preset name → preset config. */
export type PresetMap = Record<string, Preset>;

// ---------------------------------------------------------------------------
// Usage / telemetry
// ---------------------------------------------------------------------------

/**
 * Emitted via `onUsage` after every completed LLM call (including retried calls).
 * Use this to track costs, audit usage, or update analytics.
 */
export interface UsageEvent {
    /** The preset name that was used for this call. */
    preset: string;
    /** The model that actually produced the response (may differ from preset.model if a fallback was used). */
    model: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    /** Cost in USD as reported by OpenRouter. `undefined` if the API did not return cost data. */
    costUsd: number | undefined;
    /** Whether web search access was enabled for this call. */
    web: boolean;
    /** Wall-clock duration of the call from send to first parse, in milliseconds. */
    durationMs: number;
}

// ---------------------------------------------------------------------------
// OpenRouter call input shapes
// ---------------------------------------------------------------------------

/** An image to include alongside a prompt for multimodal models. */
export interface ImageInput {
    /**
     * Base64-encoded image bytes (without the `data:` URI prefix).
     * Either `base64` or `uri` must be provided.
     */
    base64?: string;
    /**
     * A remote URL or local URI that will be forwarded to the model as-is.
     * Either `base64` or `uri` must be provided.
     */
    uri?: string;
    /** MIME type, e.g. `"image/jpeg"`. Used only when `base64` is set. Defaults to `"image/jpeg"`. */
    mimeType?: string;
}

/** Parameters for a structured (JSON-schema-validated) LLM call. */
export interface StructuredParams<T extends ZodTypeAny> {
    /** Preset name to resolve model + parameters. Must exist in the preset map. */
    preset: string;
    /**
     * Zod schema describing the expected JSON output shape. Converted to a
     * JSON Schema and sent as `response_format` to enforce structured output.
     */
    schema: T;
    /**
     * Optional name for the JSON schema passed in `response_format`.
     * Defaults to a sanitized version of the preset name.
     */
    schemaName?: string;
    /** Optional system prompt to prepend before the user message. */
    system?: string;
    /** The user prompt. */
    prompt: string;
    /** Images to attach to the user message. Requires a multimodal-capable model. */
    images?: ImageInput[];
    /**
     * Grant the model access to the internet via the OpenRouter web plugin.
     * Useful for tasks that need real-time information or URL content.
     */
    web?: boolean;
    /**
     * Cancellation signal. Pass `ctx.signal` from the enclosing queued task so
     * that cancelling the task also cancels the in-flight HTTP request.
     */
    signal?: AbortSignal;
    /**
     * Maximum number of retry attempts when the model returns invalid JSON or a
     * response that does not match the schema. Default: `1`.
     */
    maxRetries?: number;
    /**
     * Per-call timeout in milliseconds. When set, a combined AbortSignal is
     * created that fires when either `timeoutMs` elapses or the provided
     * `signal` is aborted, whichever comes first.
     */
    timeoutMs?: number;
}

/** Parameters for a plain-text LLM call (no JSON schema validation). */
export interface TextParams {
    /** Preset name to resolve model + parameters. Must exist in the preset map. */
    preset: string;
    /** Optional system prompt to prepend before the user message. */
    system?: string;
    /** The user prompt. */
    prompt: string;
    /** Images to attach to the user message. Requires a multimodal-capable model. */
    images?: ImageInput[];
    /** Grant the model internet access via the OpenRouter web plugin. */
    web?: boolean;
    /**
     * Cancellation signal. Pass `ctx.signal` from the enclosing queued task so
     * that cancelling the task also cancels the in-flight HTTP request.
     */
    signal?: AbortSignal;
    /**
     * Per-call timeout in milliseconds. When set, a combined AbortSignal is
     * created that fires when either `timeoutMs` elapses or the provided
     * `signal` is aborted, whichever comes first.
     */
    timeoutMs?: number;
}

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------

/** Context injected into every queued task function. */
export interface AITaskContext {
    /**
     * AbortSignal tied to this operation's lifecycle. Pass it to `ai.structured()`,
     * `ai.text()`, and any other async work so they are cancelled together.
     */
    signal: AbortSignal;
}

/** Options controlling how an operation is scheduled and deduplicated. */
export interface AIOperationOptions {
    /**
     * Deduplication key. If an operation with the same key is already pending,
     * the newer call supersedes it (updates the task + priority) and both
     * callers share the same promise chain.
     */
    dedupeKey?: string;
    /**
     * Cancellation group identifier (e.g. an entity or session ID).
     * Calling `cancelGroup(group)` will abort all operations with this group.
     */
    group?: string;
    /**
     * Scheduling priority. Higher numbers run first. Operations at or above
     * `interactiveThreshold` (default: 90) may use burst concurrency slots.
     */
    priority?: number;
    /**
     * Optional guard called immediately before the operation starts executing.
     * Return `false` to skip the operation (it resolves as `undefined`).
     * Useful when the context that triggered the work is no longer relevant.
     */
    isStillValid?: () => boolean;
    /**
     * When `true` and the same `dedupeKey` is already running, the running
     * operation is aborted and a fresh one is started. Default: `false`
     * (new calls attach to the existing running operation instead).
     */
    restart?: boolean;
    /**
     * Mark this as a long-running / slow operation. Slow ops are capped at a
     * lower concurrency limit so at least one background slot stays free for
     * faster calls.
     */
    slowOp?: boolean;
}

/** A snapshot of the queue's current state. Returned by `getStats()`. */
export interface QueueStats {
    /** Label of the first running operation, or `null` if nothing is running. */
    activeOperation: string | null;
    /** Labels of all currently running operations. */
    runningLabels: string[];
    runningOperations: number;
    pendingOperations: number;
    /** Count of pending operations grouped by priority level. */
    pendingByPriority: Record<string, number>;
    completedOperations: number;
    failedOperations: number;
    skippedOperations: number;
    cancelledOperations: number;
    /** Aggregate token usage and estimated cost across all calls in this session. */
    usage: {
        promptTokens: number;
        completionTokens: number;
        totalTokens: number;
        estimatedCostUsd: number;
    };
}

/** Lifecycle event kinds emitted by the queue. */
export type QueueEventKind =
    | 'queued'
    | 'bumped'
    | 'started'
    | 'completed'
    | 'failed'
    | 'cancelled'
    | 'skipped';

/** Emitted to subscribers on each queue lifecycle transition. */
export interface QueueEvent {
    kind: QueueEventKind;
    /** Human-readable label of the affected operation. */
    label: string;
    group?: string;
    priority?: number;
    /** Full queue stats snapshot at the time of the event. */
    stats: QueueStats;
}

/** Callback for queue lifecycle events. Registered via `subscribe()`. */
export type QueueListener = (event: QueueEvent) => void;

// ---------------------------------------------------------------------------
// Concurrency config
// ---------------------------------------------------------------------------

/** Fine-grained concurrency limits for the operation queue. */
export interface ConcurrencyConfig {
    /**
     * Maximum number of operations allowed to run concurrently under normal
     * (non-burst) conditions. Default: `3`.
     */
    base?: number;
    /**
     * Additional concurrency slots available for urgent/interactive operations
     * (those in the focused group or at/above `interactiveThreshold`). Default: `2`.
     */
    burst?: number;
    /**
     * Maximum number of slow operations that may run concurrently. Defaults to
     * `base - 1`, ensuring at least one slot is always free for fast calls.
     */
    slowOp?: number;
}

// ---------------------------------------------------------------------------
// Main library config
// ---------------------------------------------------------------------------

/** Configuration passed to `createAIQueue()`. */
export interface CortexConfig {
    /**
     * Your OpenRouter API key. **Required.**
     * @see https://openrouter.ai/keys
     */
    apiKey: string;
    /**
     * OpenRouter API base URL.
     * Default: `"https://openrouter.ai/api/v1"`.
     */
    baseUrl?: string;
    /**
     * Preset overrides or additions, merged on top of `DEFAULT_PRESETS`.
     * Pass a map with new keys to extend the defaults, or include existing
     * keys to override them. Pass only your own keys to replace all defaults.
     */
    presets?: PresetMap;
    /** Concurrency limits. See `ConcurrencyConfig` for defaults. */
    concurrency?: ConcurrencyConfig;
    /**
     * Operations with `priority` at or above this value are considered
     * interactive and may use burst concurrency slots.
     * Default: `90`.
     */
    interactiveThreshold?: number;
    /**
     * HTTP-Referer and X-Title headers sent to OpenRouter for attribution
     * and display in the OpenRouter dashboard.
     */
    headers?: {
        referer?: string;
        title?: string;
    };
    /**
     * Logger implementation. Defaults to `noopLogger` (silent).
     * Use `consoleLogger` or `createLogger(level)` for development.
     */
    logger?: Logger;
    /**
     * Called after every completed LLM call with token counts and cost data.
     * Use this to persist usage, display running totals, or feed analytics.
     */
    onUsage?: (event: UsageEvent) => void;
}
