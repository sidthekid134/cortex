import { describe, it, expect, vi, beforeEach } from 'vitest';
import { runAgentLoop } from '../agent.js';
import { ProviderRouter, OpenRouterProvider } from '../provider.js';
import { OpenRouterClient } from '../openrouter.js';
import { AIOperationCancelledError } from '../queue.js';
import { DEFAULT_PRESETS } from '../presets.js';
import type { AgentTool, AgentToolResult, AgentToolRecord, AgentParams, UsageEvent, PresetMap } from '../types.js';
import type { ProviderChatResult } from '../provider.js';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

const PRESETS: PresetMap = DEFAULT_PRESETS;

function makeRouter(responses: ProviderChatResult[]) {
    let call = 0;
    const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
    const provider = new OpenRouterProvider(client);
    vi.spyOn(provider, 'chat').mockImplementation(async () => {
        const r = responses[call] ?? responses[responses.length - 1];
        call++;
        return r;
    });
    return new ProviderRouter(provider);
}

const DUMMY_USAGE = {
    prompt_tokens: 10,
    completion_tokens: 20,
    total_tokens: 30,
    cost: 0.001,
};

function stopResponse(content = 'Done.'): ProviderChatResult {
    return { content, toolCalls: [], finishReason: 'stop', model: 'test/m', usage: DUMMY_USAGE };
}

function toolCallResponse(calls: Array<{ id: string; name: string; args: Record<string, unknown> }>): ProviderChatResult {
    return {
        content: null,
        toolCalls: calls.map(c => ({
            id: c.id,
            type: 'function' as const,
            function: { name: c.name, arguments: JSON.stringify(c.args) },
        })),
        finishReason: 'tool_calls',
        model: 'test/m',
        usage: DUMMY_USAGE,
    };
}

function makeTool<TCtx>(
    name: string,
    fn: (args: Record<string, unknown>, ctx: TCtx) => Promise<AgentToolResult>,
    readOnly?: boolean | ((args: Record<string, unknown>) => boolean),
): AgentTool<TCtx> {
    return {
        name,
        description: `tool ${name}`,
        parameters: { type: 'object', properties: {}, required: [] },
        execute: fn,
        readOnly,
    };
}

// ---------------------------------------------------------------------------
// Happy path — single stop response
// ---------------------------------------------------------------------------

describe('runAgentLoop — happy path', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('returns text from a stop response', async () => {
        const router = makeRouter([stopResponse('Hello!')]);
        const result = await runAgentLoop(
            { preset: 'fast-vision', system: 'sys', messages: [{ role: 'user', content: 'hi' }], tools: [] },
            router,
            PRESETS,
        );
        expect(result.text).toBe('Hello!');
        expect(result.toolCalls).toHaveLength(0);
        expect(result.effects).toHaveLength(0);
    });

    it('accumulates aggregate usage across turns', async () => {
        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'get_data', args: {} }]),
            stopResponse('Done.'),
        ]);
        const tool = makeTool('get_data', async () => ({ result: 'data' }), true);
        const result = await runAgentLoop(
            { preset: 'fast-vision', system: 'sys', messages: [{ role: 'user', content: 'go' }], tools: [tool], toolContext: {} },
            router,
            PRESETS,
        );
        // Two turns × (10 prompt + 20 completion + 30 total + 0.001 cost)
        expect(result.usage.promptTokens).toBe(20);
        expect(result.usage.totalTokens).toBe(60);
        expect(result.usage.estimatedCostUsd).toBeCloseTo(0.002, 5);
    });
});

// ---------------------------------------------------------------------------
// Tool-only turn (null content)
// ---------------------------------------------------------------------------

describe('runAgentLoop — tool-only turn', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('handles null content on tool_calls turn', async () => {
        let toolCalled = false;
        const tool = makeTool('do_thing', async () => {
            toolCalled = true;
            return { result: 'done' };
        });
        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'do_thing', args: { x: 1 } }]),
            stopResponse('All set.'),
        ]);
        const result = await runAgentLoop(
            { preset: 'fast-vision', system: 'sys', messages: [{ role: 'user', content: 'go' }], tools: [tool], toolContext: {} },
            router,
            PRESETS,
        );
        expect(toolCalled).toBe(true);
        expect(result.text).toBe('All set.');
        expect(result.toolCalls).toHaveLength(1);
        expect(result.toolCalls[0].name).toBe('do_thing');
    });

    it('collects effects from tool results', async () => {
        const tool = makeTool<{}>('do_thing', async () => ({
            result: 'ok',
            effect: { type: 'test_effect' as const, value: 42 },
        }));
        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'do_thing', args: {} }]),
            stopResponse(),
        ]);
        const result = await runAgentLoop<{}, { type: 'test_effect'; value: number }>(
            { preset: 'fast-vision', system: 'sys', messages: [{ role: 'user', content: 'go' }], tools: [tool as AgentTool<{}, { type: 'test_effect'; value: number }>], toolContext: {} },
            router,
            PRESETS,
        );
        expect(result.effects).toHaveLength(1);
        expect(result.effects[0]).toEqual({ type: 'test_effect', value: 42 });
    });
});

// ---------------------------------------------------------------------------
// Parallel read batches
// ---------------------------------------------------------------------------

describe('runAgentLoop — parallel read batches', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('executes contiguous read-only tools in parallel', async () => {
        const order: string[] = [];
        const makeReadTool = (name: string, delay = 0) =>
            makeTool(name, async () => {
                await new Promise(r => setTimeout(r, delay));
                order.push(name);
                return { result: `${name}_result` };
            }, true);

        const readA = makeReadTool('read_a', 10);
        const readB = makeReadTool('read_b', 5);

        const router = makeRouter([
            toolCallResponse([
                { id: 'c1', name: 'read_a', args: {} },
                { id: 'c2', name: 'read_b', args: {} },
            ]),
            stopResponse(),
        ]);

        await runAgentLoop(
            { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [readA, readB], toolContext: {} },
            router,
            PRESETS,
        );

        // read_b has shorter delay — if parallel, it finishes first
        expect(order).toEqual(['read_b', 'read_a']);
    });

    it('memoises identical read-only calls across turns', async () => {
        let callCount = 0;
        const readTool = makeTool('get_data', async () => {
            callCount++;
            return { result: 'cached_val' };
        }, true);

        // Two turns: both request get_data with the same args.
        // Turn 1 runs the tool and caches the result.
        // Turn 2 should find the cached result and not re-execute.
        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'get_data', args: { key: 'x' } }]),
            toolCallResponse([{ id: 'c2', name: 'get_data', args: { key: 'x' } }]),
            stopResponse(),
        ]);

        const result = await runAgentLoop(
            { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [readTool], toolContext: {} },
            router,
            PRESETS,
        );

        // Turn 2 uses cache from turn 1 — tool only executed once total
        expect(callCount).toBe(1);
        expect(result.toolCalls).toHaveLength(2);
        expect(result.toolCalls[1].result).toBe('cached_val');
    });
});

// ---------------------------------------------------------------------------
// Serial write ordering
// ---------------------------------------------------------------------------

describe('runAgentLoop — serial write ordering', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('executes write tools in LLM-emitted order', async () => {
        const order: string[] = [];
        const makeWriteTool = (name: string) =>
            makeTool(name, async () => {
                order.push(name);
                return { result: `${name}_done` };
            });

        const router = makeRouter([
            toolCallResponse([
                { id: 'c1', name: 'write_a', args: {} },
                { id: 'c2', name: 'write_b', args: {} },
                { id: 'c3', name: 'write_c', args: {} },
            ]),
            stopResponse(),
        ]);

        await runAgentLoop(
            {
                preset: 'fast-vision',
                system: 's',
                messages: [{ role: 'user', content: 'go' }],
                tools: [makeWriteTool('write_a'), makeWriteTool('write_b'), makeWriteTool('write_c')],
                toolContext: {},
            },
            router,
            PRESETS,
        );

        expect(order).toEqual(['write_a', 'write_b', 'write_c']);
    });

    it('refreshes toolContext before each write tool', async () => {
        const states = [{ value: 0 }, { value: 1 }];
        let stateIdx = 0;
        const getToolContext = vi.fn(() => states[stateIdx++] ?? states[states.length - 1]);

        const seenContexts: number[] = [];
        const tool = makeTool<{ value: number }>('write_thing', async (_, ctx) => {
            seenContexts.push(ctx.value);
            return { result: 'ok' };
        });

        const router = makeRouter([
            toolCallResponse([
                { id: 'c1', name: 'write_thing', args: {} },
                { id: 'c2', name: 'write_thing', args: {} },
            ]),
            stopResponse(),
        ]);

        await runAgentLoop(
            { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [tool], getToolContext },
            router,
            PRESETS,
        );

        // Each write gets a fresh context
        expect(seenContexts).toEqual([0, 1]);
        expect(getToolContext).toHaveBeenCalledTimes(2);
    });
});

// ---------------------------------------------------------------------------
// Callable readOnly function (dispatcher-style action classification)
// ---------------------------------------------------------------------------

describe('runAgentLoop — function-based readOnly', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('classifies at action level when readOnly is a function', async () => {
        const order: string[] = [];
        const dispatcher = makeTool(
            'dispatch',
            async (args) => {
                order.push(args.action as string);
                return { result: 'ok' };
            },
            (args) => args.action === 'read',
        );

        // read, read (parallel), then write (serial)
        const router = makeRouter([
            toolCallResponse([
                { id: 'c1', name: 'dispatch', args: { action: 'read' } },
                { id: 'c2', name: 'dispatch', args: { action: 'read' } },
                { id: 'c3', name: 'dispatch', args: { action: 'write' } },
            ]),
            stopResponse(),
        ]);

        await runAgentLoop(
            { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [dispatcher], toolContext: {} },
            router,
            PRESETS,
        );

        // Both reads ran (order unspecified), then write
        expect(order).toContain('read');
        expect(order[order.length - 1]).toBe('write');
    });
});

// ---------------------------------------------------------------------------
// Cancellation via signal
// ---------------------------------------------------------------------------

describe('runAgentLoop — cancellation', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('aborts when signal is already aborted before start', async () => {
        const controller = new AbortController();
        controller.abort(new AIOperationCancelledError('test cancel'));
        const router = makeRouter([stopResponse()]);

        await expect(
            runAgentLoop(
                { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [], signal: controller.signal },
                router,
                PRESETS,
            ),
        ).rejects.toBeInstanceOf(AIOperationCancelledError);
    });

    it('aborts mid-loop when signal fires between turns', async () => {
        const controller = new AbortController();
        let routerCallCount = 0;

        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const provider = new OpenRouterProvider(client);
        vi.spyOn(provider, 'chat').mockImplementation(async () => {
            routerCallCount++;
            if (routerCallCount === 1) {
                // Abort after first turn returns
                controller.abort(new AIOperationCancelledError('cancelled'));
            }
            return toolCallResponse([{ id: 'c1', name: 'some_tool', args: {} }]);
        });
        const router = new ProviderRouter(provider);

        const tool = makeTool('some_tool', async () => ({ result: 'ok' }));

        await expect(
            runAgentLoop(
                { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [tool], toolContext: {}, signal: controller.signal },
                router,
                PRESETS,
            ),
        ).rejects.toBeInstanceOf(AIOperationCancelledError);
    });
});

// ---------------------------------------------------------------------------
// Max-turns abort
// ---------------------------------------------------------------------------

describe('runAgentLoop — max turns', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('stops after maxTurns and returns accumulated text', async () => {
        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'loop_tool', args: {} }]),
        ]);
        const tool = makeTool('loop_tool', async () => ({ result: 'keep going' }));

        const result = await runAgentLoop(
            {
                preset: 'fast-vision',
                system: 's',
                messages: [{ role: 'user', content: 'go' }],
                tools: [tool],
                toolContext: {},
                maxTurns: 3,
            },
            router,
            PRESETS,
        );

        // Loop should have stopped after 3 turns
        expect(result.text).toBeTruthy();
        expect(result.toolCalls.length).toBeLessThanOrEqual(3);
    });
});

// ---------------------------------------------------------------------------
// All-tool-failure abort
// ---------------------------------------------------------------------------

describe('runAgentLoop — consecutive failure abort', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('aborts after 3 consecutive all-failure turns', async () => {
        const tool = makeTool('bad_tool', async () => {
            throw new Error('always fails');
        });

        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'bad_tool', args: {} }]),
        ]);

        const result = await runAgentLoop(
            { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [tool], toolContext: {}, maxTurns: 20 },
            router,
            PRESETS,
        );

        // Should have stopped due to consecutive failures, not max turns
        expect(result.text).toContain('Tool execution failed for "bad_tool"');
        expect(result.toolCalls.length).toBeLessThanOrEqual(3);
    });
});

// ---------------------------------------------------------------------------
// onToolBatch hook
// ---------------------------------------------------------------------------

describe('runAgentLoop — onToolBatch', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('calls onToolBatch after each tool execution batch', async () => {
        const batches: AgentToolRecord[][] = [];
        const tool = makeTool('action', async () => ({ result: 'done' }));

        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'action', args: { x: 1 } }]),
            stopResponse(),
        ]);

        await runAgentLoop(
            {
                preset: 'fast-vision',
                system: 's',
                messages: [{ role: 'user', content: 'go' }],
                tools: [tool],
                toolContext: {},
                onToolBatch: async (batch) => { batches.push(batch); },
            },
            router,
            PRESETS,
        );

        expect(batches).toHaveLength(1);
        expect(batches[0][0].name).toBe('action');
        expect(batches[0][0].args).toEqual({ x: 1 });
        expect(batches[0][0].result).toBe('done');
    });
});

// ---------------------------------------------------------------------------
// Unknown tool
// ---------------------------------------------------------------------------

describe('runAgentLoop — unknown tool', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('returns error string for unknown tool without throwing', async () => {
        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'nonexistent', args: {} }]),
            stopResponse(),
        ]);

        const result = await runAgentLoop(
            { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [], toolContext: {} },
            router,
            PRESETS,
        );

        expect(result.toolCalls[0].result).toBe('Unknown tool: nonexistent');
    });
});

// ---------------------------------------------------------------------------
// Per-turn onUsage emission
// ---------------------------------------------------------------------------

describe('runAgentLoop — per-turn onUsage', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('emits onUsage for each LLM turn', async () => {
        const usageEvents: UsageEvent[] = [];
        const tool = makeTool('act', async () => ({ result: 'ok' }));

        const router = makeRouter([
            toolCallResponse([{ id: 'c1', name: 'act', args: {} }]),
            stopResponse('done'),
        ]);

        await runAgentLoop(
            { preset: 'fast-vision', system: 's', messages: [{ role: 'user', content: 'go' }], tools: [tool], toolContext: {} },
            router,
            PRESETS,
            (e) => usageEvents.push(e),
        );

        expect(usageEvents).toHaveLength(2);
        expect(usageEvents[0].preset).toBe('fast-vision');
        expect(usageEvents[0].promptTokens).toBe(10);
        expect(usageEvents[1].completionTokens).toBe(20);
    });
});

// ---------------------------------------------------------------------------
// ProviderRouter capability routing
// ---------------------------------------------------------------------------

describe('ProviderRouter — capability routing', () => {
    beforeEach(() => vi.restoreAllMocks());

    it('falls back to cloud when preferred provider lacks tools', async () => {
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const cloudProvider = new OpenRouterProvider(client);
        const cloudChat = vi.spyOn(cloudProvider, 'chat').mockResolvedValue(stopResponse('cloud'));

        const router = new ProviderRouter(cloudProvider);

        // Register a provider that does NOT support tools
        const limitedProvider = {
            id: 'limited',
            capabilities: { tools: false, vision: false, structuredOutput: false },
            async chat() { return stopResponse('limited'); },
        };
        router.registerProvider(limitedProvider);

        const { ProviderRouter: _, ...rest } = await import('../provider.js');

        const result = await router.route(
            {
                preset: DEFAULT_PRESETS['fast-vision']!,
                messages: [{ role: 'user', content: 'hi' }],
                tools: [{ type: 'function', function: { name: 'do_thing', parameters: {} } }],
            },
            'limited',
        );

        expect(cloudChat).toHaveBeenCalledOnce();
        expect(result.content).toBe('cloud');
    });

    it('falls back to cloud when preferred provider is unavailable', async () => {
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const cloudProvider = new OpenRouterProvider(client);
        vi.spyOn(cloudProvider, 'chat').mockResolvedValue(stopResponse('cloud'));

        const router = new ProviderRouter(cloudProvider);

        const unavailableProvider = {
            id: 'offline',
            capabilities: { tools: true, vision: false, structuredOutput: false },
            isAvailable: () => false,
            async chat() { return stopResponse('offline'); },
        };
        router.registerProvider(unavailableProvider);

        const result = await router.route(
            { preset: DEFAULT_PRESETS['fast-vision']!, messages: [{ role: 'user', content: 'hi' }] },
            'offline',
        );

        expect(result.content).toBe('cloud');
    });

    it('uses preferred provider when available and capable', async () => {
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const cloudProvider = new OpenRouterProvider(client);
        vi.spyOn(cloudProvider, 'chat').mockResolvedValue(stopResponse('cloud'));

        const router = new ProviderRouter(cloudProvider);

        const customProvider = {
            id: 'custom',
            capabilities: { tools: true, vision: true, structuredOutput: true },
            isAvailable: () => true,
            async chat() { return stopResponse('custom'); },
        };
        router.registerProvider(customProvider);

        const result = await router.route(
            { preset: DEFAULT_PRESETS['fast-vision']!, messages: [{ role: 'user', content: 'hi' }] },
            'custom',
        );

        expect(result.content).toBe('custom');
    });
});

// ---------------------------------------------------------------------------
// chatWithTools() on OpenRouterClient
// ---------------------------------------------------------------------------

describe('OpenRouterClient.chatWithTools()', () => {
    beforeEach(() => vi.restoreAllMocks());

    function mockFetch(body: unknown, status = 200) {
        vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
            ok: status >= 200 && status < 300,
            status,
            statusText: status === 200 ? 'OK' : 'Error',
            json: async () => body,
            text: async () => JSON.stringify(body),
        }));
    }

    it('returns stop response with text content', async () => {
        mockFetch({
            choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'hello', tool_calls: [] } }],
            model: 'test/m',
            usage: DUMMY_USAGE,
        });
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const result = await client.chatWithTools(
            DEFAULT_PRESETS['fast-vision']!,
            [{ role: 'user', content: 'hi' }],
        );
        expect(result.finishReason).toBe('stop');
        expect(result.content).toBe('hello');
        expect(result.toolCalls).toHaveLength(0);
    });

    it('returns tool_calls response with null content', async () => {
        const wireToolCall = {
            id: 'c1',
            type: 'function',
            function: { name: 'do_thing', arguments: '{"x":1}' },
        };
        mockFetch({
            choices: [{
                finish_reason: 'tool_calls',
                message: { role: 'assistant', content: null, tool_calls: [wireToolCall] },
            }],
            model: 'test/m',
            usage: DUMMY_USAGE,
        });
        const client = new OpenRouterClient({ apiKey: 'k', baseUrl: 'http://x' });
        const result = await client.chatWithTools(
            DEFAULT_PRESETS['fast-vision']!,
            [{ role: 'user', content: 'hi' }],
            { tools: [{ type: 'function', function: { name: 'do_thing', description: 'test', parameters: {} } }] },
        );
        expect(result.finishReason).toBe('tool_calls');
        expect(result.content).toBeNull();
        expect(result.toolCalls).toHaveLength(1);
        expect(result.toolCalls[0].function.name).toBe('do_thing');
    });
});
