import { z } from 'zod';
import type { ZodTypeAny } from 'zod';
import type { StructuredParams, TextParams, UsageEvent, PresetMap, Logger } from './types.js';
import {
    OpenRouterClient,
    buildUserContent,
    type ChatMessage,
    type ResponseFormat,
} from './openrouter.js';
import { noopLogger } from './logger.js';
import type { AIOperationQueue } from './queue.js';
import { recordCallUsage } from './telemetry.js';
import { isAIAbortError, AIOperationCancelledError } from './queue.js';

// ---------------------------------------------------------------------------
// AbortSignal helpers
// ---------------------------------------------------------------------------

/**
 * RN-safe replacement for `AbortSignal.prototype.throwIfAborted()`, which is
 * not implemented in React Native / Hermes. Throws the signal's abort reason
 * (or a generic cancellation error) when the signal is already aborted.
 */
function throwIfAborted(signal: AbortSignal | undefined): void {
    if (!signal?.aborted) return;
    const reason = (signal as AbortSignal & { reason?: unknown }).reason;
    if (reason instanceof Error) throw reason;
    throw new AIOperationCancelledError(
        typeof reason === 'string' ? reason : undefined,
    );
}

// ---------------------------------------------------------------------------
// StructuredOutputError
// ---------------------------------------------------------------------------

/**
 * Thrown by `structured()` when all retry attempts are exhausted without
 * producing a valid response. Inspect `cause` for the last parse or
 * schema-validation error.
 */
export class StructuredOutputError extends Error {
    /**
     * The parse or Zod validation error from the final attempt.
     * Use `instanceof ZodError` to access validation details.
     */
    readonly cause: unknown;
    /** Number of attempts made (including the first try). */
    readonly attempts: number;
    /** Preset name that was used. */
    readonly preset: string;

    constructor(
        message: string,
        options: { cause: unknown; attempts: number; preset: string },
    ) {
        super(message);
        this.name = 'StructuredOutputError';
        this.cause = options.cause;
        this.attempts = options.attempts;
        this.preset = options.preset;
    }
}

// ---------------------------------------------------------------------------
// Timeout signal helper
// ---------------------------------------------------------------------------

/**
 * Wraps an optional signal and optional timeout into a single AbortSignal.
 * Returns the original signal unchanged when no timeout is needed, avoiding
 * unnecessary allocations on the hot path.
 */
function withTimeout(
    signal: AbortSignal | undefined,
    timeoutMs: number | undefined,
): { signal: AbortSignal | undefined; cleanup: () => void } {
    if (!timeoutMs) return { signal, cleanup: () => {} };

    const controller = new AbortController();

    // Fire after timeoutMs
    const timerId = setTimeout(() => {
        controller.abort(new Error(`[cortex] call timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    // Forward cancellation from the parent signal, if any
    if (signal) {
        if (signal.aborted) {
            clearTimeout(timerId);
            controller.abort(signal.reason);
        } else {
            signal.addEventListener(
                'abort',
                () => {
                    clearTimeout(timerId);
                    controller.abort(signal.reason);
                },
                { once: true },
            );
        }
    }

    return {
        signal: controller.signal,
        cleanup: () => clearTimeout(timerId),
    };
}

// ---------------------------------------------------------------------------
// structured()
// ---------------------------------------------------------------------------

/**
 * Make a structured LLM call: sends the prompt, requests JSON output conforming
 * to the provided Zod schema, validates the response, and retries on
 * schema-mismatch or parse failure. Returns the typed, validated result.
 *
 * Pass `ctx.signal` from the queue task so cancellation propagates to the
 * in-flight HTTP request. Aborted calls are never retried.
 *
 * @throws {StructuredOutputError} when all retries are exhausted.
 * @throws {AIOperationCancelledError} when the signal is aborted.
 * @throws {OpenRouterError} on non-retryable HTTP errors.
 */
export async function structured<T extends ZodTypeAny>(
    params: StructuredParams<T>,
    presets: PresetMap,
    client: OpenRouterClient,
    queue: AIOperationQueue,
    onUsage?: (e: UsageEvent) => void,
    logger: Logger = noopLogger,
): Promise<z.infer<T>> {
    const preset = resolvePreset(params.preset, presets, logger);
    const schemaName = params.schemaName ?? sanitizeName(params.preset);
    const jsonSchema = buildJsonSchema(params.schema);
    const responseFormat: ResponseFormat = {
        type: 'json_schema',
        json_schema: { name: schemaName, schema: jsonSchema, strict: true },
    };

    const messages = buildMessages(params.system, params.prompt, params.images);
    const maxRetries = params.maxRetries ?? 1;
    const totalAttempts = maxRetries + 1;

    const { signal: callSignal, cleanup } = withTimeout(params.signal, params.timeoutMs);

    let lastError: unknown;
    const startedAt = Date.now();

    logger.debug(`structured "${params.preset}" — ${totalAttempts} attempt(s) max`);

    try {
        for (let attempt = 0; attempt < totalAttempts; attempt++) {
            throwIfAborted(callSignal);

            if (attempt > 0) {
                const reason =
                    lastError instanceof Error ? lastError.message : String(lastError);
                logger.warn(
                    `structured "${params.preset}" — retry ${attempt}/${maxRetries}: ${reason}`,
                );
            }

            const callMessages =
                attempt === 0
                    ? messages
                    : appendRetryMessage(messages, lastError);

            let result;
            try {
                result = await client.chat(preset, callMessages, {
                    responseFormat,
                    web: params.web,
                    signal: callSignal,
                });
            } catch (err) {
                // Never retry on abort — propagate immediately.
                if (isAIAbortError(err) || (callSignal?.aborted ?? false)) throw err;
                // Non-retryable HTTP errors are also propagated immediately.
                lastError = err;
                logger.warn(
                    `structured "${params.preset}" — network error on attempt ${attempt + 1}: ${err instanceof Error ? err.message : String(err)}`,
                );
                continue;
            }

            const parsed = tryParseJSON(result.content);
            if (!parsed.ok) {
                lastError = new Error(`Invalid JSON from model: ${parsed.raw}`);
                logger.warn(
                    `structured "${params.preset}" — invalid JSON on attempt ${attempt + 1}: ${parsed.raw}`,
                );
                continue;
            }

            const validated = params.schema.safeParse(parsed.value);
            if (!validated.success) {
                lastError = validated.error;
                logger.warn(
                    `structured "${params.preset}" — schema mismatch on attempt ${attempt + 1}: ${validated.error.message}`,
                );
                continue;
            }

            recordCallUsage(
                queue,
                {
                    preset: params.preset,
                    model: result.model,
                    usage: result.usage,
                    web: params.web ?? false,
                    durationMs: Date.now() - startedAt,
                },
                onUsage,
            );

            logger.debug(
                `structured "${params.preset}" — success on attempt ${attempt + 1}`,
            );
            return validated.data as z.infer<T>;
        }
    } finally {
        cleanup();
    }

    logger.error(
        `structured "${params.preset}" — all ${totalAttempts} attempt(s) failed`,
        lastError,
    );
    throw new StructuredOutputError(
        `[cortex] structured("${params.preset}") failed after ${totalAttempts} attempt(s)`,
        { cause: lastError, attempts: totalAttempts, preset: params.preset },
    );
}

// ---------------------------------------------------------------------------
// text()
// ---------------------------------------------------------------------------

/**
 * Make a plain-text LLM call (no JSON schema, no validation). Useful for
 * free-form generation where the app post-processes the result itself.
 *
 * @throws {AIOperationCancelledError} when the signal is aborted.
 * @throws {OpenRouterError} on HTTP errors.
 */
export async function text(
    params: TextParams,
    presets: PresetMap,
    client: OpenRouterClient,
    queue: AIOperationQueue,
    onUsage?: (e: UsageEvent) => void,
    logger: Logger = noopLogger,
): Promise<string> {
    const preset = resolvePreset(params.preset, presets, logger);
    const messages = buildMessages(params.system, params.prompt, params.images);
    const startedAt = Date.now();

    const { signal: callSignal, cleanup } = withTimeout(params.signal, params.timeoutMs);

    logger.debug(`text "${params.preset}"`);

    try {
        throwIfAborted(callSignal);

        const result = await client.chat(preset, messages, {
            web: params.web,
            signal: callSignal,
        });

        recordCallUsage(
            queue,
            {
                preset: params.preset,
                model: result.model,
                usage: result.usage,
                web: params.web ?? false,
                durationMs: Date.now() - startedAt,
            },
            onUsage,
        );

        return result.content;
    } finally {
        cleanup();
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

function resolvePreset(name: string, presets: PresetMap, logger: Logger) {
    const preset = presets[name];
    if (!preset) {
        const available = Object.keys(presets).join(', ');
        logger.error(`Unknown preset "${name}". Available: ${available}`);
        throw new Error(
            `[cortex] Unknown preset "${name}". Available: ${available}`,
        );
    }
    return preset;
}

/**
 * Convert a Zod schema to a JSON Schema object compatible with OpenRouter's
 * `response_format`. Strips the `$schema` meta-key so the API doesn't reject it.
 */
function buildJsonSchema(schema: ZodTypeAny): Record<string, unknown> {
    const full = z.toJSONSchema(schema) as Record<string, unknown>;
    const { $schema: _drop, ...rest } = full;
    return rest;
}

function buildMessages(
    system: string | undefined,
    prompt: string,
    images?: StructuredParams<ZodTypeAny>['images'],
): ChatMessage[] {
    const messages: ChatMessage[] = [];
    if (system) messages.push({ role: 'system', content: system });
    messages.push({ role: 'user', content: buildUserContent(prompt, images) });
    return messages;
}

function appendRetryMessage(messages: ChatMessage[], error: unknown): ChatMessage[] {
    const detail = error instanceof Error ? error.message : String(error);
    return [
        ...messages,
        {
            role: 'user',
            content: `Your previous response could not be parsed. Please respond with valid JSON matching the schema exactly. Error: ${detail}`,
        },
    ];
}

type JSONParseResult =
    | { ok: true; value: unknown }
    | { ok: false; raw: string };

function tryParseJSON(raw: string): JSONParseResult {
    const trimmed = raw.trim();
    // Strip markdown code fences in case the model wraps output in ```json ... ```
    const unwrapped = trimmed.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
    try {
        return { ok: true, value: JSON.parse(unwrapped) };
    } catch {
        return { ok: false, raw: trimmed.slice(0, 200) };
    }
}

function sanitizeName(name: string): string {
    return name.replace(/[^a-zA-Z0-9_-]/g, '_');
}
