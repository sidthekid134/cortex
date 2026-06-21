import type { ImageInput, Preset, Logger } from './types.js';
import { noopLogger } from './logger.js';

// ---------------------------------------------------------------------------
// OpenRouter wire types
// ---------------------------------------------------------------------------

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant';
    content: string | ContentPart[];
}

export type ContentPart = TextPart | ImageUrlPart;

export interface TextPart {
    type: 'text';
    text: string;
}

export interface ImageUrlPart {
    type: 'image_url';
    image_url: { url: string };
}

export interface ResponseFormat {
    type: 'json_schema';
    json_schema: {
        name: string;
        schema: Record<string, unknown>;
        strict: boolean;
    };
}

export interface ChatRequest {
    model: string;
    models?: string[];
    messages: ChatMessage[];
    temperature?: number;
    max_tokens?: number;
    response_format?: ResponseFormat;
    plugins?: Array<{ id: string }>;
    usage?: { include: true };
}

export interface ChatUsage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    /** OpenRouter-reported cost in USD (present when `usage.include = true`). */
    cost?: number;
}

export interface ChatResponse {
    choices: Array<{
        message: {
            content: string | null;
            role: string;
        };
    }>;
    model?: string;
    usage?: ChatUsage;
}

export interface ChatResult {
    content: string;
    /** The model that actually responded (may differ from preset.model if a fallback was used). */
    model: string;
    usage: ChatUsage;
}

// ---------------------------------------------------------------------------
// Client config
// ---------------------------------------------------------------------------

export interface OpenRouterClientConfig {
    apiKey: string;
    baseUrl: string;
    referer?: string | undefined;
    title?: string | undefined;
    logger?: Logger | undefined;
}

// ---------------------------------------------------------------------------
// OpenRouter client
// ---------------------------------------------------------------------------

export class OpenRouterClient {
    private readonly headers: Record<string, string>;
    private readonly baseUrl: string;
    private readonly logger: Logger;

    constructor(cfg: OpenRouterClientConfig) {
        this.baseUrl = cfg.baseUrl.replace(/\/$/, '');
        this.logger = cfg.logger ?? noopLogger;
        this.headers = {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${cfg.apiKey}`,
            ...(cfg.referer ? { 'HTTP-Referer': cfg.referer } : {}),
            ...(cfg.title ? { 'X-Title': cfg.title } : {}),
        };
    }

    async chat(
        preset: Preset,
        messages: ChatMessage[],
        options?: {
            responseFormat?: ResponseFormat;
            web?: boolean;
            signal?: AbortSignal;
        },
    ): Promise<ChatResult> {
        const body: ChatRequest = {
            model: preset.model,
            ...(preset.fallbacks?.length ? { models: [preset.model, ...preset.fallbacks] } : {}),
            messages,
            ...(preset.temperature !== undefined ? { temperature: preset.temperature } : {}),
            ...(preset.maxTokens !== undefined ? { max_tokens: preset.maxTokens } : {}),
            ...(options?.responseFormat ? { response_format: options.responseFormat } : {}),
            ...(options?.web ? { plugins: [{ id: 'web' }] } : {}),
            usage: { include: true },
        };

        const fallbackNote = preset.fallbacks?.length
            ? ` (+${preset.fallbacks.length} fallback${preset.fallbacks.length > 1 ? 's' : ''})`
            : '';
        this.logger.debug(
            `→ ${preset.model}${fallbackNote} · ${messages.length} msg(s)${options?.web ? ' · web' : ''}${options?.responseFormat ? ' · json_schema' : ''}`,
        );

        const startedAt = Date.now();
        const response = await fetch(`${this.baseUrl}/chat/completions`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(body),
            signal: options?.signal,
        });

        if (!response.ok) {
            let detail = '';
            try { detail = await response.text(); } catch { /* ignore */ }
            const isRetryable = response.status === 429 || response.status >= 500;
            this.logger.error(
                `← HTTP ${response.status} ${response.statusText}${isRetryable ? ' (retryable)' : ''}${detail ? ` — ${detail.slice(0, 300)}` : ''}`,
            );
            throw new OpenRouterError(
                `OpenRouter ${response.status}: ${response.statusText}${detail ? ` — ${detail}` : ''}`,
                response.status,
                isRetryable,
            );
        }

        const data = (await response.json()) as ChatResponse;
        const choice = data.choices?.[0];
        if (!choice) {
            throw new OpenRouterError('OpenRouter returned no choices', 0, false);
        }

        const content = choice.message.content;
        if (content === null || content === undefined) {
            throw new OpenRouterError('OpenRouter returned null content', 0, false);
        }

        const durationMs = Date.now() - startedAt;
        const usage = data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
        this.logger.debug(
            `← ${data.model ?? preset.model} · ${usage.total_tokens} tokens (↑${usage.prompt_tokens} ↓${usage.completion_tokens})${usage.cost !== undefined ? ` · $${usage.cost.toFixed(6)}` : ''} · ${durationMs}ms`,
        );

        return {
            content,
            model: data.model ?? preset.model,
            usage,
        };
    }
}

// ---------------------------------------------------------------------------
// OpenRouterError
// ---------------------------------------------------------------------------

/**
 * Thrown when the OpenRouter API returns a non-2xx response or an
 * unexpected payload (e.g. empty choices).
 */
export class OpenRouterError extends Error {
    /**
     * HTTP status code from the OpenRouter response.
     * `0` indicates a non-HTTP error (e.g. empty response body).
     */
    readonly status: number;
    /**
     * `true` when the request can safely be retried.
     * Retryable statuses: 429 (rate limit) and 5xx (server errors).
     */
    readonly isRetryable: boolean;

    constructor(message: string, status: number, isRetryable: boolean) {
        super(message);
        this.name = 'OpenRouterError';
        this.status = status;
        this.isRetryable = isRetryable;
    }
}

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

/**
 * Convert an `ImageInput` to an OpenRouter `image_url` content part.
 * Throws if neither `base64` nor `uri` is supplied.
 */
export function imageToContentPart(image: ImageInput): ImageUrlPart {
    if (image.base64) {
        const mime = image.mimeType ?? 'image/jpeg';
        return { type: 'image_url', image_url: { url: `data:${mime};base64,${image.base64}` } };
    }
    if (image.uri) {
        return { type: 'image_url', image_url: { url: image.uri } };
    }
    throw new Error('ImageInput must have either base64 or uri');
}

/**
 * Build a user message content array with optional images prepended before
 * the prompt text. Returns a plain string when no images are provided.
 */
export function buildUserContent(prompt: string, images?: ImageInput[]): string | ContentPart[] {
    if (!images?.length) return prompt;
    return [
        ...images.map(imageToContentPart),
        { type: 'text', text: prompt } satisfies TextPart,
    ];
}
