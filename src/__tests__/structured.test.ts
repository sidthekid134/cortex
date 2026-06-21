import { describe, it, expect, beforeEach, vi } from 'vitest';
import { z } from 'zod';
import { structured, text } from '../structured.js';
import { AIOperationQueue } from '../queue.js';
import { OpenRouterClient } from '../openrouter.js';
import type { PresetMap, UsageEvent } from '../types.js';
import { DEFAULT_PRESETS } from '../presets.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TestSchema = z.object({
    name: z.string(),
    value: z.number(),
});

type TestType = z.infer<typeof TestSchema>;

const VALID_RESPONSE: TestType = { name: 'foo', value: 42 };

function makeQueue() {
    return new AIOperationQueue({ base: 3 }, 90);
}

function makePresets(overrides?: PresetMap): PresetMap {
    return { ...DEFAULT_PRESETS, ...overrides };
}

function mockChat(responses: Array<string | Error>) {
    let call = 0;
    return vi.spyOn(OpenRouterClient.prototype, 'chat').mockImplementation(async () => {
        const resp = responses[call++] ?? responses[responses.length - 1];
        if (resp instanceof Error) throw resp;
        return {
            content: typeof resp === 'string' ? resp : JSON.stringify(resp),
            model: 'test/model',
            usage: { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, cost: 0.001 },
        };
    });
}

// ---------------------------------------------------------------------------
// structured() — happy path
// ---------------------------------------------------------------------------

describe('structured() — happy path', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('returns typed result from valid JSON response', async () => {
        mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        const result = await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test' },
            makePresets(),
            client,
            q,
        );

        expect(result).toEqual(VALID_RESPONSE);
    });

    it('strips ```json code fences before parsing', async () => {
        mockChat(['```json\n{"name":"bar","value":7}\n```']);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        const result = await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test' },
            makePresets(),
            client,
            q,
        );
        expect(result).toEqual({ name: 'bar', value: 7 });
    });

    it('strips plain ``` code fences', async () => {
        mockChat(['```\n{"name":"baz","value":3}\n```']);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        const result = await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test' },
            makePresets(),
            client,
            q,
        );
        expect(result).toEqual({ name: 'baz', value: 3 });
    });

    it('passes system message when provided', async () => {
        const spy = mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await structured(
            { preset: 'fast-vision', schema: TestSchema, system: 'Be precise.', prompt: 'test' },
            makePresets(),
            client,
            q,
        );

        const messages = spy.mock.calls[0][1];
        expect(messages[0]).toMatchObject({ role: 'system', content: 'Be precise.' });
    });

    it('uses response_format json_schema', async () => {
        const spy = mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test' },
            makePresets(),
            client,
            q,
        );

        const opts = spy.mock.calls[0][2];
        expect(opts?.responseFormat?.type).toBe('json_schema');
        expect(opts?.responseFormat?.json_schema.name).toBeTruthy();
        expect(opts?.responseFormat?.json_schema.strict).toBe(true);
    });

    it('passes web: true option through', async () => {
        const spy = mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test', web: true },
            makePresets(),
            client,
            q,
        );

        expect(spy.mock.calls[0][2]?.web).toBe(true);
    });

    it('threads AbortSignal to the chat call', async () => {
        const spy = mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const controller = new AbortController();

        await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test', signal: controller.signal },
            makePresets(),
            client,
            q,
        );

        expect(spy.mock.calls[0][2]?.signal).toBe(controller.signal);
    });
});

// ---------------------------------------------------------------------------
// structured() — retry logic
// ---------------------------------------------------------------------------

describe('structured() — retry', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('retries once on invalid JSON and succeeds', async () => {
        const spy = mockChat(['not valid json', JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        const result = await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test', maxRetries: 1 },
            makePresets(),
            client,
            q,
        );

        expect(result).toEqual(VALID_RESPONSE);
        expect(spy).toHaveBeenCalledTimes(2);
    });

    it('retries once on schema-validation failure and succeeds', async () => {
        const invalid = JSON.stringify({ name: 'ok', value: 'not-a-number' });
        const valid = JSON.stringify(VALID_RESPONSE);
        const spy = mockChat([invalid, valid]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        const result = await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test', maxRetries: 1 },
            makePresets(),
            client,
            q,
        );

        expect(result).toEqual(VALID_RESPONSE);
        expect(spy).toHaveBeenCalledTimes(2);
    });

    it('appends a corrective message on retry', async () => {
        const spy = mockChat(['bad json', JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'original', maxRetries: 1 },
            makePresets(),
            client,
            q,
        );

        const retryMessages = spy.mock.calls[1][1];
        const lastMsg = retryMessages[retryMessages.length - 1];
        expect(lastMsg.role).toBe('user');
        expect((lastMsg.content as string).toLowerCase()).toContain('previous response');
    });

    it('throws after all retries are exhausted', async () => {
        mockChat(['bad', 'also bad']);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await expect(
            structured(
                { preset: 'fast-vision', schema: TestSchema, prompt: 'test', maxRetries: 1 },
                makePresets(),
                client,
                q,
            ),
        ).rejects.toThrow();
    });

    it('does not retry when maxRetries is 0', async () => {
        const spy = mockChat(['bad json', JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await expect(
            structured(
                { preset: 'fast-vision', schema: TestSchema, prompt: 'test', maxRetries: 0 },
                makePresets(),
                client,
                q,
            ),
        ).rejects.toThrow();

        expect(spy).toHaveBeenCalledTimes(1);
    });
});

// ---------------------------------------------------------------------------
// structured() — telemetry
// ---------------------------------------------------------------------------

describe('structured() — telemetry', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('calls onUsage with correct data after success', async () => {
        mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const usageEvents: UsageEvent[] = [];

        await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test' },
            makePresets(),
            client,
            q,
            (e) => usageEvents.push(e),
        );

        expect(usageEvents).toHaveLength(1);
        expect(usageEvents[0].preset).toBe('fast-vision');
        expect(usageEvents[0].promptTokens).toBe(10);
        expect(usageEvents[0].completionTokens).toBe(20);
        expect(usageEvents[0].costUsd).toBe(0.001);
        expect(usageEvents[0].durationMs).toBeGreaterThanOrEqual(0);
    });

    it('records usage into queue stats', async () => {
        mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test' },
            makePresets(),
            client,
            q,
        );

        const { usage } = q.getStats();
        expect(usage.promptTokens).toBe(10);
        expect(usage.completionTokens).toBe(20);
        expect(usage.totalTokens).toBe(30);
        expect(usage.estimatedCostUsd).toBe(0.001);
    });

    it('does not call onUsage when retries exhausted (no success)', async () => {
        mockChat(['bad', 'still bad']);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const usageEvents: UsageEvent[] = [];

        await structured(
            { preset: 'fast-vision', schema: TestSchema, prompt: 'test', maxRetries: 1 },
            makePresets(),
            client,
            q,
            (e) => usageEvents.push(e),
        ).catch(() => {});

        expect(usageEvents).toHaveLength(0);
    });
});

// ---------------------------------------------------------------------------
// structured() — error cases
// ---------------------------------------------------------------------------

describe('structured() — error cases', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('throws on unknown preset', async () => {
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await expect(
            structured(
                { preset: 'nonexistent-preset', schema: TestSchema, prompt: 'test' },
                makePresets(),
                client,
                q,
            ),
        ).rejects.toThrow('nonexistent-preset');
    });

    it('propagates AbortError when signal is already aborted', async () => {
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const controller = new AbortController();
        controller.abort();

        await expect(
            structured(
                { preset: 'fast-vision', schema: TestSchema, prompt: 'test', signal: controller.signal },
                makePresets(),
                client,
                q,
            ),
        ).rejects.toThrow();
    });

    it('propagates error from the underlying chat call', async () => {
        vi.spyOn(OpenRouterClient.prototype, 'chat').mockRejectedValue(new Error('network error'));
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await expect(
            structured(
                { preset: 'fast-vision', schema: TestSchema, prompt: 'test' },
                makePresets(),
                client,
                q,
            ),
        ).rejects.toThrow('network error');
    });
});

// ---------------------------------------------------------------------------
// text() helper
// ---------------------------------------------------------------------------

describe('text()', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('returns the raw string content', async () => {
        vi.spyOn(OpenRouterClient.prototype, 'chat').mockResolvedValue({
            content: 'raw text response',
            model: 'test/model',
            usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15 },
        });

        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        const result = await text(
            { preset: 'cheap-text', prompt: 'say hello' },
            makePresets(),
            client,
            q,
        );

        expect(result).toBe('raw text response');
    });

    it('calls onUsage after success', async () => {
        vi.spyOn(OpenRouterClient.prototype, 'chat').mockResolvedValue({
            content: 'hi',
            model: 'test/model',
            usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15, cost: 0.0005 },
        });

        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const usageEvents: UsageEvent[] = [];

        await text(
            { preset: 'cheap-text', prompt: 'say hello' },
            makePresets(),
            client,
            q,
            (e) => usageEvents.push(e),
        );

        expect(usageEvents).toHaveLength(1);
        expect(usageEvents[0].preset).toBe('cheap-text');
        expect(usageEvents[0].costUsd).toBe(0.0005);
    });

    it('throws on unknown preset', async () => {
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await expect(
            text({ preset: 'nope', prompt: 'hi' }, makePresets(), client, q),
        ).rejects.toThrow('nope');
    });

    it('does not send response_format', async () => {
        const spy = vi.spyOn(OpenRouterClient.prototype, 'chat').mockResolvedValue({
            content: 'hi',
            model: 'test/model',
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        });

        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        await text({ preset: 'cheap-text', prompt: 'hi' }, makePresets(), client, q);

        const opts = spy.mock.calls[0][2];
        expect(opts?.responseFormat).toBeUndefined();
    });
});

// ---------------------------------------------------------------------------
// multimodal image content
// ---------------------------------------------------------------------------

describe('structured() — multimodal images', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('includes image parts in user message when images provided', async () => {
        const spy = mockChat([JSON.stringify(VALID_RESPONSE)]);
        const q = makeQueue();
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });

        await structured(
            {
                preset: 'fast-vision',
                schema: TestSchema,
                prompt: 'what is in this image?',
                images: [{ uri: 'https://example.com/photo.jpg' }],
            },
            makePresets(),
            client,
            q,
        );

        const messages = spy.mock.calls[0][1];
        const userMsg = messages.find((m) => m.role === 'user');
        expect(Array.isArray(userMsg?.content)).toBe(true);
        const parts = userMsg?.content as Array<{ type: string }>;
        expect(parts.some((p) => p.type === 'image_url')).toBe(true);
    });
});
