import { z } from 'zod';
import type { ZodTypeAny } from 'zod';
import type { StructuredParams, TextParams, UsageEvent, PresetMap } from './types.js';
import {
    OpenRouterClient,
    buildUserContent,
    type ChatMessage,
    type ResponseFormat,
} from './openrouter.js';
import type { AIOperationQueue } from './queue.js';
import { recordCallUsage } from './telemetry.js';

// ---------------------------------------------------------------------------
// Structured output call
// ---------------------------------------------------------------------------

/**
 * Make a structured LLM call: sends the prompt, requests JSON output conforming
 * to the provided Zod schema, validates the response, and retries once on
 * schema-mismatch. Returns the typed, validated result.
 *
 * Pass `ctx.signal` from the queue task so cancellation propagates.
 */
export async function structured<T extends ZodTypeAny>(
    params: StructuredParams<T>,
    presets: PresetMap,
    client: OpenRouterClient,
    queue: AIOperationQueue,
    onUsage?: (e: UsageEvent) => void,
): Promise<z.infer<T>> {
    const preset = resolvePreset(params.preset, presets);
    const schemaName = params.schemaName ?? sanitizeName(params.preset);
    const jsonSchema = buildJsonSchema(params.schema);
    const responseFormat: ResponseFormat = {
        type: 'json_schema',
        json_schema: { name: schemaName, schema: jsonSchema, strict: true },
    };

    const messages = buildMessages(params.system, params.prompt, params.images);
    const maxRetries = params.maxRetries ?? 1;

    let lastError: unknown;
    const startedAt = Date.now();

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
        params.signal?.throwIfAborted();

        const callMessages =
            attempt === 0
                ? messages
                : appendRetryMessage(messages, lastError);

        const result = await client.chat(preset, callMessages, {
            responseFormat,
            web: params.web,
            signal: params.signal,
        });

        const parsed = tryParseJSON(result.content);
        if (!parsed.ok) {
            lastError = new Error(`Invalid JSON from model: ${parsed.raw}`);
            continue;
        }

        const validated = params.schema.safeParse(parsed.value);
        if (!validated.success) {
            lastError = validated.error;
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

        return validated.data as z.infer<T>;
    }

    throw lastError;
}

// ---------------------------------------------------------------------------
// Plain text call
// ---------------------------------------------------------------------------

/**
 * Make a plain-text LLM call (no JSON schema, no validation). Useful for
 * free-form generation where the app post-processes the result itself.
 */
export async function text(
    params: TextParams,
    presets: PresetMap,
    client: OpenRouterClient,
    queue: AIOperationQueue,
    onUsage?: (e: UsageEvent) => void,
): Promise<string> {
    const preset = resolvePreset(params.preset, presets);
    const messages = buildMessages(params.system, params.prompt, params.images);
    const startedAt = Date.now();

    params.signal?.throwIfAborted();

    const result = await client.chat(preset, messages, {
        web: params.web,
        signal: params.signal,
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
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resolvePreset(name: string, presets: PresetMap) {
    const preset = presets[name];
    if (!preset) throw new Error(`[cortex] Unknown preset "${name}". Available: ${Object.keys(presets).join(', ')}`);
    return preset;
}

/**
 * Convert a Zod schema to a plain JSON Schema object.
 * Strips the $schema meta-key so OpenRouter doesn't reject it.
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
    // Strip markdown code fences if the model wraps output in ```json ... ```
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
