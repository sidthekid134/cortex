import type { ZodTypeAny } from 'zod';

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

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
 * A named model configuration. Apps pick a preset name per operation; the
 * library resolves the actual model + params to use.
 */
export interface Preset {
    /** Primary model (OpenRouter model string, e.g. "google/gemini-flash-1.5"). */
    model: string;
    /**
     * Optional fallback chain. OpenRouter will try each in order when the
     * primary errors. Pass as `models` array in the request.
     */
    fallbacks?: string[];
    temperature?: number;
    maxTokens?: number;
}

export type PresetMap = Record<string, Preset>;

// ---------------------------------------------------------------------------
// Usage / telemetry
// ---------------------------------------------------------------------------

/** Emitted after every completed call (including retries). */
export interface UsageEvent {
    /** The preset name that was used. */
    preset: string;
    /** Resolved model that actually responded. */
    model: string;
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    /** Cost in USD as reported by OpenRouter (undefined if not returned). */
    costUsd: number | undefined;
    /** Whether web access was requested for this call. */
    web: boolean;
    durationMs: number;
}

// ---------------------------------------------------------------------------
// OpenRouter call shapes
// ---------------------------------------------------------------------------

export interface ImageInput {
    /** base64-encoded image data (without data: prefix). */
    base64?: string;
    /** Remote or local URI that will be sent as a URL to the model. */
    uri?: string;
    mimeType?: string;
}

export interface StructuredParams<T extends ZodTypeAny> {
    preset: string;
    schema: T;
    /** Optional name for the JSON schema (used in response_format). */
    schemaName?: string;
    system?: string;
    prompt: string;
    images?: ImageInput[];
    /** Allow the model to search the internet. Maps to OpenRouter web plugin. */
    web?: boolean;
    signal?: AbortSignal;
    /** Max times to retry if the model returns invalid JSON/schema. Default: 1. */
    maxRetries?: number;
}

export interface TextParams {
    preset: string;
    system?: string;
    prompt: string;
    images?: ImageInput[];
    web?: boolean;
    signal?: AbortSignal;
}

// ---------------------------------------------------------------------------
// Queue
// ---------------------------------------------------------------------------

/** Context handed to every queued task. */
export interface AITaskContext {
    signal: AbortSignal;
}

export interface AIOperationOptions {
    /** Dedupe key — a newer enqueue with the same key supersedes the pending one. */
    dedupeKey?: string;
    /** Cancellation group (e.g. the entity/session id this work belongs to). */
    group?: string;
    priority?: number;
    /** Checked right before the task runs; if false the task is skipped. */
    isStillValid?: () => boolean;
    /**
     * If the same dedupeKey is already running, abort it and start fresh
     * instead of attaching to the existing run.
     */
    restart?: boolean;
    /**
     * Mark as a slow/long-running call. Slow ops are capped at a lower
     * concurrency so at least one background slot stays free for fast calls.
     */
    slowOp?: boolean;
}

export interface QueueStats {
    activeOperation: string | null;
    runningOperations: number;
    pendingOperations: number;
    pendingByPriority: Record<string, number>;
    completedOperations: number;
    failedOperations: number;
    skippedOperations: number;
    cancelledOperations: number;
    /** Aggregate usage across all completed calls in this session. */
    usage: {
        promptTokens: number;
        completionTokens: number;
        totalTokens: number;
        estimatedCostUsd: number;
    };
}

export type QueueEventKind =
    | 'queued'
    | 'bumped'
    | 'started'
    | 'completed'
    | 'failed'
    | 'cancelled'
    | 'skipped';

export interface QueueEvent {
    kind: QueueEventKind;
    label: string;
    group?: string;
    priority?: number;
    stats: QueueStats;
}

export type QueueListener = (event: QueueEvent) => void;

// ---------------------------------------------------------------------------
// Concurrency config
// ---------------------------------------------------------------------------

export interface ConcurrencyConfig {
    /** Baseline concurrent operations. Default: 3. */
    base?: number;
    /** Extra slots for urgent/focused work. Default: 2. */
    burst?: number;
    /** Max concurrent slow ops in background. Default: base - 1. */
    slowOp?: number;
}

// ---------------------------------------------------------------------------
// Main library config
// ---------------------------------------------------------------------------

export interface CortexConfig {
    /** OpenRouter API key. */
    apiKey: string;
    /** Default: "https://openrouter.ai/api/v1". */
    baseUrl?: string;
    /**
     * Preset overrides / additions merged over DEFAULT_PRESETS.
     * Pass a full map to replace defaults entirely, or add keys to extend.
     */
    presets?: PresetMap;
    concurrency?: ConcurrencyConfig;
    /**
     * Operations at/above this priority are "urgent" and may use burst slots.
     * Default: 90.
     */
    interactiveThreshold?: number;
    /** OpenRouter HTTP-Referer + X-Title headers for attribution. */
    headers?: {
        referer?: string;
        title?: string;
    };
    logger?: Logger;
    /** Called after every completed LLM call with cost + token data. */
    onUsage?: (event: UsageEvent) => void;
}
