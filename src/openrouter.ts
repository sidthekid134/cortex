import type { ImageInput, Preset, Logger } from './types.js';
import { noopLogger } from './logger.js';

// ---------------------------------------------------------------------------
// OpenRouter wire types
// ---------------------------------------------------------------------------

// Tool spec sent to the model (OpenAI / OpenRouter function-calling format).
export interface ToolSpec {
    type: 'function';
    function: {
        name: string;
        description?: string;
        parameters: Record<string, unknown>;
    };
}

export type ToolChoice =
    | 'auto'
    | 'none'
    | 'required'
    | { type: 'function'; function: { name: string } };

// A single tool call emitted by the model in an assistant message.
export interface ToolCallWire {
    id: string;
    type: 'function';
    function: {
        name: string;
        /** JSON-encoded argument object. */
        arguments: string;
    };
}

export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    /** null is valid for assistant messages that contain only tool_calls. */
    content: string | ContentPart[] | null;
    /** Tool calls emitted by the model (assistant role only). */
    tool_calls?: ToolCallWire[];
    /** Ties a tool result back to the original tool call (tool role only). */
    tool_call_id?: string;
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
    tools?: ToolSpec[];
    tool_choice?: ToolChoice;
    parallel_tool_calls?: boolean;
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
        finish_reason?: string;
        message: {
            content: string | null;
            role: string;
            tool_calls?: ToolCallWire[];
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

/** Result returned by chatWithTools() — supports both text and tool-call responses. */
export interface ToolChatResult {
    /** Text content from the model, or null when the response contains only tool calls. */
    content: string | null;
    toolCalls: ToolCallWire[];
    finishReason: 'stop' | 'tool_calls' | 'length' | 'content_filter';
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
        // Store the raw key — we validate it at call time so that a missing key
        // never throws synchronously during module initialisation (which can corrupt
        // the Hermes GC in React Native's new architecture and cause a SIGSEGV).
        this.headers = {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${cfg.apiKey ?? ''}`,
            ...(cfg.referer ? { 'HTTP-Referer': cfg.referer } : {}),
            ...(cfg.title ? { 'X-Title': cfg.title } : {}),
        };
    }

    private assertApiKey(): void {
        if (this.headers['Authorization'] === 'Bearer ') {
            throw new Error(
                '[cortex] OPENROUTER_API_KEY is missing. ' +
                'Set it in your .env file and rebuild the app.',
            );
        }
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
        this.assertApiKey();
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

    /**
     * Make a tool-aware chat request. Unlike `chat()`, this method:
     * - Sends tool definitions and tool-call results using the OpenAI function-calling wire format.
     * - Accepts assistant messages with null content (model emitted only tool calls).
     * - Returns the finish reason and any tool calls emitted by the model.
     *
     * Use this inside an agent loop. For structured output without tools, use `chat()` instead.
     */
    async chatWithTools(
        preset: Preset,
        messages: ChatMessage[],
        options?: {
            tools?: ToolSpec[];
            toolChoice?: ToolChoice;
            parallelToolCalls?: boolean;
            signal?: AbortSignal;
        },
    ): Promise<ToolChatResult> {
        this.assertApiKey();
        const body: ChatRequest = {
            model: preset.model,
            ...(preset.fallbacks?.length ? { models: [preset.model, ...preset.fallbacks] } : {}),
            messages,
            ...(preset.temperature !== undefined ? { temperature: preset.temperature } : {}),
            ...(preset.maxTokens !== undefined ? { max_tokens: preset.maxTokens } : {}),
            usage: { include: true },
            ...(options?.tools?.length ? { tools: options.tools } : {}),
            ...(options?.toolChoice !== undefined ? { tool_choice: options.toolChoice } : {}),
            ...(options?.parallelToolCalls !== undefined
                ? { parallel_tool_calls: options.parallelToolCalls }
                : {}),
        };

        const toolNote = options?.tools?.length ? ` · ${options.tools.length} tool(s)` : '';
        const fallbackNote = preset.fallbacks?.length
            ? ` (+${preset.fallbacks.length} fallback${preset.fallbacks.length > 1 ? 's' : ''})`
            : '';
        this.logger.debug(`→ ${preset.model}${fallbackNote}${toolNote} · ${messages.length} msg(s)`);

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

        const durationMs = Date.now() - startedAt;
        const usage = data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
        const toolCalls = choice.message.tool_calls ?? [];
        const rawFinish = choice.finish_reason ?? 'stop';
        const finishReason = normalizeFinishReason(rawFinish);

        this.logger.debug(
            `← ${data.model ?? preset.model} · ${usage.total_tokens} tokens (↑${usage.prompt_tokens} ↓${usage.completion_tokens})${usage.cost !== undefined ? ` · $${usage.cost.toFixed(6)}` : ''} · ${durationMs}ms · ${finishReason}${toolCalls.length ? ` · ${toolCalls.length} tool call(s)` : ''}`,
        );

        return {
            content: choice.message.content ?? null,
            toolCalls,
            finishReason,
            model: data.model ?? preset.model,
            usage,
        };
    }
}

function normalizeFinishReason(raw: string): ToolChatResult['finishReason'] {
    if (raw === 'tool_calls') return 'tool_calls';
    if (raw === 'length' || raw === 'max_tokens') return 'length';
    if (raw === 'content_filter') return 'content_filter';
    return 'stop';
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
