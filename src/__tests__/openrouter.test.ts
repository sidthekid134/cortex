import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
    OpenRouterClient,
    OpenRouterError,
    imageToContentPart,
    buildUserContent,
} from '../openrouter.js';
import type { Preset } from '../types.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DEFAULT_PRESET: Preset = {
    model: 'test/model',
    temperature: 0.5,
    maxTokens: 512,
};

const PRESET_WITH_FALLBACKS: Preset = {
    model: 'test/primary',
    fallbacks: ['test/fallback1', 'test/fallback2'],
};

function makeClient(overrides?: { referer?: string; title?: string }) {
    return new OpenRouterClient({
        apiKey: 'test-key',
        baseUrl: 'https://openrouter.ai/api/v1',
        ...overrides,
    });
}

function mockFetch(body: unknown, status = 200) {
    vi.stubGlobal(
        'fetch',
        vi.fn().mockResolvedValue({
            ok: status >= 200 && status < 300,
            status,
            statusText: status === 200 ? 'OK' : 'Error',
            json: async () => body,
            text: async () => JSON.stringify(body),
        }),
    );
}

function successResponse(content: string, model = 'test/model') {
    return {
        choices: [{ message: { content, role: 'assistant' } }],
        model,
        usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, cost: 0.001 },
    };
}

// ---------------------------------------------------------------------------
// Request construction
// ---------------------------------------------------------------------------

describe('OpenRouterClient.chat — request construction', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('sends POST to /chat/completions', async () => {
        mockFetch(successResponse('hello'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const [url] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
        expect(url).toBe('https://openrouter.ai/api/v1/chat/completions');
        expect((vi.mocked(fetch).mock.calls[0][1] as RequestInit).method).toBe('POST');
    });

    it('includes Authorization header', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit;
        expect((init.headers as Record<string, string>)['Authorization']).toBe('Bearer test-key');
    });

    it('includes HTTP-Referer and X-Title when provided', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient({ referer: 'https://myapp.com', title: 'MyApp' });
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const headers = vi.mocked(fetch).mock.calls[0][1]?.headers as Record<string, string>;
        expect(headers['HTTP-Referer']).toBe('https://myapp.com');
        expect(headers['X-Title']).toBe('MyApp');
    });

    it('omits HTTP-Referer and X-Title when not provided', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const headers = vi.mocked(fetch).mock.calls[0][1]?.headers as Record<string, string>;
        expect(headers['HTTP-Referer']).toBeUndefined();
        expect(headers['X-Title']).toBeUndefined();
    });

    it('always sends usage: { include: true }', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const body = JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string);
        expect(body.usage).toEqual({ include: true });
    });

    it('sends temperature and max_tokens from preset', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const body = JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string);
        expect(body.temperature).toBe(0.5);
        expect(body.max_tokens).toBe(512);
    });

    it('sends models array when preset has fallbacks', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(PRESET_WITH_FALLBACKS, [{ role: 'user', content: 'hi' }]);

        const body = JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string);
        expect(body.models).toEqual(['test/primary', 'test/fallback1', 'test/fallback2']);
    });

    it('omits models array when no fallbacks', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const body = JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string);
        expect(body.models).toBeUndefined();
    });

    it('adds web plugin when web: true', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }], { web: true });

        const body = JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string);
        expect(body.plugins).toEqual([{ id: 'web' }]);
    });

    it('omits plugins when web is not set', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        const body = JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string);
        expect(body.plugins).toBeUndefined();
    });

    it('sends response_format when provided', async () => {
        mockFetch(successResponse('{"name":"test"}'));
        const client = makeClient();
        const responseFormat = {
            type: 'json_schema' as const,
            json_schema: { name: 'TestSchema', schema: { type: 'object' }, strict: true },
        };
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }], { responseFormat });

        const body = JSON.parse((vi.mocked(fetch).mock.calls[0][1] as RequestInit).body as string);
        expect(body.response_format).toEqual(responseFormat);
    });

    it('threads AbortSignal to fetch', async () => {
        mockFetch(successResponse('hi'));
        const client = makeClient();
        const controller = new AbortController();
        await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }], {
            signal: controller.signal,
        });

        const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit;
        expect(init.signal).toBe(controller.signal);
    });
});

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

describe('OpenRouterClient.chat — response parsing', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('returns content, model, and usage', async () => {
        mockFetch(successResponse('hello world', 'actual/model'));
        const client = makeClient();
        const result = await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);

        expect(result.content).toBe('hello world');
        expect(result.model).toBe('actual/model');
        expect(result.usage.prompt_tokens).toBe(10);
        expect(result.usage.completion_tokens).toBe(20);
        expect(result.usage.cost).toBe(0.001);
    });

    it('falls back to preset model when response omits model', async () => {
        const resp = successResponse('hi');
        const { model: _drop, ...withoutModel } = resp;
        mockFetch(withoutModel);

        const client = makeClient();
        const result = await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);
        expect(result.model).toBe('test/model');
    });

    it('uses zero usage when response omits usage', async () => {
        const resp = successResponse('hi');
        const { usage: _drop, ...withoutUsage } = resp;
        mockFetch(withoutUsage);

        const client = makeClient();
        const result = await client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]);
        expect(result.usage.prompt_tokens).toBe(0);
        expect(result.usage.total_tokens).toBe(0);
    });
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe('OpenRouterClient.chat — errors', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('throws OpenRouterError on non-OK status', async () => {
        mockFetch({ error: 'unauthorized' }, 401);
        const client = makeClient();
        await expect(
            client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]),
        ).rejects.toBeInstanceOf(OpenRouterError);
    });

    it('includes status code in OpenRouterError', async () => {
        mockFetch({ error: 'rate limited' }, 429);
        const client = makeClient();
        const err = await client
            .chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }])
            .catch((e) => e);
        expect(err.status).toBe(429);
    });

    it('throws when choices array is empty', async () => {
        mockFetch({ choices: [], usage: {} });
        const client = makeClient();
        await expect(
            client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]),
        ).rejects.toBeInstanceOf(OpenRouterError);
    });

    it('throws when message content is null', async () => {
        mockFetch({
            choices: [{ message: { content: null, role: 'assistant' } }],
            usage: {},
        });
        const client = makeClient();
        await expect(
            client.chat(DEFAULT_PRESET, [{ role: 'user', content: 'hi' }]),
        ).rejects.toBeInstanceOf(OpenRouterError);
    });
});

// ---------------------------------------------------------------------------
// imageToContentPart
// ---------------------------------------------------------------------------

describe('imageToContentPart', () => {
    it('converts base64 to data URL image_url part', () => {
        const part = imageToContentPart({ base64: 'abc123', mimeType: 'image/png' });
        expect(part.type).toBe('image_url');
        expect(part.image_url.url).toBe('data:image/png;base64,abc123');
    });

    it('defaults mimeType to image/jpeg for base64', () => {
        const part = imageToContentPart({ base64: 'abc123' });
        expect(part.image_url.url).toContain('image/jpeg');
    });

    it('converts uri to image_url part', () => {
        const part = imageToContentPart({ uri: 'https://example.com/photo.jpg' });
        expect(part.type).toBe('image_url');
        expect(part.image_url.url).toBe('https://example.com/photo.jpg');
    });

    it('throws when neither base64 nor uri provided', () => {
        expect(() => imageToContentPart({})).toThrow();
    });
});

// ---------------------------------------------------------------------------
// buildUserContent
// ---------------------------------------------------------------------------

describe('buildUserContent', () => {
    it('returns a plain string when no images', () => {
        const result = buildUserContent('hello');
        expect(result).toBe('hello');
    });

    it('returns a plain string when images array is empty', () => {
        const result = buildUserContent('hello', []);
        expect(result).toBe('hello');
    });

    it('returns a content array with images first, then text', () => {
        const result = buildUserContent('describe this', [
            { uri: 'https://example.com/img.jpg' },
        ]) as Array<{ type: string }>;

        expect(Array.isArray(result)).toBe(true);
        expect(result[0].type).toBe('image_url');
        expect(result[result.length - 1].type).toBe('text');
        expect((result[result.length - 1] as { type: string; text: string }).text).toBe('describe this');
    });

    it('handles multiple images', () => {
        const result = buildUserContent('prompt', [
            { uri: 'https://a.com/1.jpg' },
            { uri: 'https://a.com/2.jpg' },
        ]) as Array<{ type: string }>;

        const imageParts = result.filter((p) => p.type === 'image_url');
        const textParts = result.filter((p) => p.type === 'text');
        expect(imageParts).toHaveLength(2);
        expect(textParts).toHaveLength(1);
    });
});
