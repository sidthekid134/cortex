import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createAIQueue } from '../createAIQueue.js';
import { consoleLogger } from '../logger.js';
import { z } from 'zod';
import { OpenRouterClient } from '../openrouter.js';

function makeAI(overrides?: Partial<Parameters<typeof createAIQueue>[0]>) {
    return createAIQueue({
        apiKey: 'test-key',
        headers: { referer: 'https://test.app', title: 'Test' },
        ...overrides,
    });
}

// ---------------------------------------------------------------------------
// Factory / surface
// ---------------------------------------------------------------------------

describe('createAIQueue', () => {
    it('returns an object with all expected methods', () => {
        const ai = makeAI();
        expect(typeof ai.enqueue).toBe('function');
        expect(typeof ai.setFocusedGroup).toBe('function');
        expect(typeof ai.cancelGroup).toBe('function');
        expect(typeof ai.cancelKey).toBe('function');
        expect(typeof ai.hasPending).toBe('function');
        expect(typeof ai.getStats).toBe('function');
        expect(typeof ai.subscribe).toBe('function');
        expect(typeof ai.drain).toBe('function');
        expect(typeof ai.destroy).toBe('function');
        expect(typeof ai.structured).toBe('function');
        expect(typeof ai.text).toBe('function');
    });

    it('getStats() returns initial zero state', () => {
        const ai = makeAI();
        const stats = ai.getStats();
        expect(stats.runningOperations).toBe(0);
        expect(stats.pendingOperations).toBe(0);
        expect(stats.completedOperations).toBe(0);
        expect(stats.failedOperations).toBe(0);
        expect(stats.usage.totalTokens).toBe(0);
        expect(stats.usage.estimatedCostUsd).toBe(0);
    });

    it('subscribe returns an unsubscribe function', () => {
        const ai = makeAI();
        const unsub = ai.subscribe(() => {});
        expect(typeof unsub).toBe('function');
        unsub();
    });

    it('accepts custom concurrency config', () => {
        const ai = createAIQueue({
            apiKey: 'k',
            concurrency: { base: 5, burst: 1, slowOp: 2 },
        });
        expect(ai.getStats().runningOperations).toBe(0);
    });

    it('accepts custom presets that extend defaults', () => {
        const ai = createAIQueue({
            apiKey: 'k',
            presets: { 'my-preset': { model: 'custom/model', temperature: 0.1 } },
        });
        // Should not throw — we just verify the instance was created successfully
        expect(ai).toBeDefined();
    });

    it('works with consoleLogger without throwing', async () => {
        const ai = createAIQueue({ apiKey: 'k', logger: consoleLogger });
        await ai.enqueue('test', async () => 'done');
    });

    it('drain() resolves when queue is empty', async () => {
        const ai = makeAI();
        await expect(ai.drain()).resolves.toBeUndefined();
    });

    it('destroy() prevents further enqueue calls', async () => {
        const ai = makeAI();
        ai.destroy();
        await expect(ai.enqueue('after', async () => {})).rejects.toThrow();
    });

    it('getStats() includes runningLabels', () => {
        const ai = makeAI();
        const stats = ai.getStats();
        expect(Array.isArray(stats.runningLabels)).toBe(true);
        expect(stats.runningLabels).toHaveLength(0);
    });

    it('calls onUsage when a structured call succeeds', async () => {
        vi.spyOn(OpenRouterClient.prototype, 'chat').mockResolvedValue({
            content: JSON.stringify({ name: 'test', value: 1 }),
            model: 'test/model',
            usage: { prompt_tokens: 5, completion_tokens: 10, total_tokens: 15, cost: 0.001 },
        });

        const usageEvents: unknown[] = [];
        const ai = makeAI({ onUsage: (e) => usageEvents.push(e) });

        const schema = z.object({ name: z.string(), value: z.number() });
        await ai.structured({ preset: 'fast-vision', schema, prompt: 'test' });

        expect(usageEvents).toHaveLength(1);
        vi.restoreAllMocks();
    });
});

// ---------------------------------------------------------------------------
// Config validation
// ---------------------------------------------------------------------------

describe('createAIQueue — config validation', () => {
    it('throws when apiKey is missing', () => {
        expect(() => createAIQueue({ apiKey: '' })).toThrow('apiKey');
    });

    it('throws when concurrency.base is not a positive integer', () => {
        expect(() => createAIQueue({ apiKey: 'k', concurrency: { base: 0 } })).toThrow('concurrency.base');
        expect(() => createAIQueue({ apiKey: 'k', concurrency: { base: 1.5 } })).toThrow('concurrency.base');
    });

    it('throws when concurrency.burst is negative', () => {
        expect(() => createAIQueue({ apiKey: 'k', concurrency: { burst: -1 } })).toThrow('concurrency.burst');
    });

    it('accepts valid concurrency values', () => {
        expect(() => createAIQueue({ apiKey: 'k', concurrency: { base: 5, burst: 0, slowOp: 2 } })).not.toThrow();
    });
});

// ---------------------------------------------------------------------------
// Integration: enqueue + structured
// ---------------------------------------------------------------------------

describe('enqueue + structured integration', () => {
    beforeEach(() => { vi.restoreAllMocks(); });

    it('runs a structured call inside a queued task', async () => {
        vi.spyOn(OpenRouterClient.prototype, 'chat').mockResolvedValue({
            content: '{"result":true}',
            model: 'test/model',
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        });

        const ai = makeAI();
        const schema = z.object({ result: z.boolean() });

        const result = await ai.enqueue('my-task', async (ctx) => {
            return ai.structured({ preset: 'fast-vision', schema, prompt: 'test', signal: ctx.signal });
        });

        expect(result.result).toBe(true);
        expect(ai.getStats().completedOperations).toBe(1);
    });

    it('cancelGroup cancels the queued task before it runs', async () => {
        const ai = makeAI({ concurrency: { base: 1 } });
        const blockD = { resolve: () => {} } as { resolve: () => void };
        const blocker = new Promise<void>((r) => { blockD.resolve = r; });

        // Keep the blocker promise handled so we don't get unhandled rejection
        const blockerPromise = ai.enqueue('blocker', () => blocker, { group: 'g-block' });
        await new Promise<void>((r) => setTimeout(r, 0));

        const p = ai.enqueue('task', async () => 'done', { group: 'my-group' });
        ai.cancelGroup('my-group');

        await expect(p).rejects.toThrow();

        // Clean up the blocker
        blockD.resolve();
        await blockerPromise;
    });
});

// ---------------------------------------------------------------------------
// Custom baseUrl
// ---------------------------------------------------------------------------

describe('custom baseUrl', () => {
    it('uses the provided baseUrl', async () => {
        vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
            ok: true,
            status: 200,
            json: async () => ({
                choices: [{ message: { content: '"hi"', role: 'assistant' } }],
                model: 'test/model',
                usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            }),
        }));

        const ai = createAIQueue({
            apiKey: 'k',
            baseUrl: 'https://custom.api.com/v2',
        });
        const schema = z.string();
        await ai.structured({ preset: 'fast-vision', schema, prompt: 'hi' });

        const [url] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit];
        expect(url).toContain('https://custom.api.com/v2');

        vi.restoreAllMocks();
    });
});
