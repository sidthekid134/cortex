import type { ImageInput, Preset } from './types.js';

// ---------------------------------------------------------------------------
// OpenRouter chat completion — low-level fetch client
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
    /** OpenRouter-reported cost in USD (present when usage.include = true). */
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
    model: string;
    usage: ChatUsage;
}

// ---------------------------------------------------------------------------
// OpenRouter client factory
// ---------------------------------------------------------------------------

export interface OpenRouterClientConfig {
    apiKey: string;
    baseUrl: string;
    referer?: string | undefined;
    title?: string | undefined;
}

export class OpenRouterClient {
    private readonly headers: Record<string, string>;
    private readonly baseUrl: string;

    constructor(cfg: OpenRouterClientConfig) {
        this.baseUrl = cfg.baseUrl.replace(/\/$/, '');
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

        const response = await fetch(`${this.baseUrl}/chat/completions`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify(body),
            signal: options?.signal,
        });

        if (!response.ok) {
            let detail = '';
            try { detail = await response.text(); } catch { /* ignore */ }
            throw new OpenRouterError(
                `OpenRouter ${response.status}: ${response.statusText}${detail ? ` — ${detail}` : ''}`,
                response.status,
            );
        }

        const data = (await response.json()) as ChatResponse;
        const choice = data.choices?.[0];
        if (!choice) throw new OpenRouterError('OpenRouter returned no choices', 0);

        const content = choice.message.content;
        if (content === null || content === undefined) {
            throw new OpenRouterError('OpenRouter returned null content', 0);
        }

        return {
            content,
            model: data.model ?? preset.model,
            usage: data.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        };
    }
}

export class OpenRouterError extends Error {
    constructor(
        message: string,
        public readonly status: number,
    ) {
        super(message);
        this.name = 'OpenRouterError';
    }
}

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

/** Convert an ImageInput to an OpenRouter image_url content part. */
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

/** Build the user content array with optional images prepended before the prompt text. */
export function buildUserContent(prompt: string, images?: ImageInput[]): string | ContentPart[] {
    if (!images?.length) return prompt;
    return [
        ...images.map(imageToContentPart),
        { type: 'text', text: prompt } satisfies TextPart,
    ];
}
