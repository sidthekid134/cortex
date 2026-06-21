/**
 * Provider-agnostic multi-turn tool-calling agent loop.
 *
 * Drives the chat → tool_calls → execute → feed_results → repeat cycle.
 * Domain-specific knowledge lives entirely in the caller-supplied `tools` and
 * lifecycle hooks (`getToolContext`, `onToolBatch`).
 *
 * Designed to run as a single `enqueue()` task so scheduling, cancellation,
 * priority, and deduplication are managed by the Cortex queue:
 *
 * ```ts
 * ai.enqueue('coach:recommend', async (ctx) =>
 *   ai.agent({ preset: 'coaching', system, messages, tools, signal: ctx.signal }),
 *   { group: sessionId, dedupeKey: 'coaching:recommend', priority: 40 }
 * );
 * ```
 *
 * Inner per-turn LLM calls are made directly through the ProviderRouter (not
 * re-enqueued) to avoid self-deadlock on concurrency limits.
 */

import type {
    AgentParams,
    AgentResult,
    AgentTool,
    AgentToolRecord,
    AgentToolResult,
    PresetMap,
    UsageEvent,
    Logger,
} from './types.js';
import type { ChatMessage, ToolSpec, ToolCallWire } from './openrouter.js';
import type { ProviderRouter } from './provider.js';
import { AIOperationCancelledError } from './queue.js';
import { noopLogger } from './logger.js';

// ---------------------------------------------------------------------------
// AbortSignal helper (RN-safe, same pattern as structured.ts)
// ---------------------------------------------------------------------------

function throwIfAborted(signal: AbortSignal | undefined): void {
    if (!signal?.aborted) return;
    const reason = (signal as AbortSignal & { reason?: unknown }).reason;
    if (reason instanceof Error) throw reason;
    throw new AIOperationCancelledError(
        typeof reason === 'string' ? reason : undefined,
    );
}

// ---------------------------------------------------------------------------
// Preset resolver (shared pattern with structured.ts)
// ---------------------------------------------------------------------------

function resolvePreset(name: string, presets: PresetMap, logger: Logger) {
    const preset = presets[name];
    if (!preset) {
        const available = Object.keys(presets).join(', ');
        logger.error(`[cortex] agent: unknown preset "${name}". Available: ${available}`);
        throw new Error(
            `[cortex] agent: unknown preset "${name}". Available: ${available}`,
        );
    }
    return preset;
}

// ---------------------------------------------------------------------------
// readOnly classifier
// ---------------------------------------------------------------------------

function isToolReadOnly(
    tool: AgentTool,
    args: Record<string, unknown>,
): boolean {
    if (tool.readOnly === undefined || tool.readOnly === false) return false;
    if (tool.readOnly === true) return true;
    return tool.readOnly(args);
}

// ---------------------------------------------------------------------------
// Safe JSON parse
// ---------------------------------------------------------------------------

function parseArgs(raw: string, toolName: string, logger: Logger): Record<string, unknown> {
    try {
        return JSON.parse(raw) as Record<string, unknown>;
    } catch {
        logger.warn(`[cortex] agent: failed to parse args for "${toolName}" — treating as empty`);
        return {};
    }
}

// ---------------------------------------------------------------------------
// Core loop
// ---------------------------------------------------------------------------

const DEFAULT_MAX_TURNS = 12;
const MAX_CONSECUTIVE_ALL_FAILURE_TURNS = 3;

export async function runAgentLoop<TContext, TEffect>(
    params: AgentParams<TContext, TEffect>,
    router: ProviderRouter,
    presets: PresetMap,
    onUsage?: (e: UsageEvent) => void,
    logger: Logger = noopLogger,
): Promise<AgentResult<TEffect>> {
    const {
        preset: presetName,
        provider: preferredProvider,
        system,
        messages,
        tools,
        getToolContext,
        toolContext,
        onToolBatch,
        maxTurns = DEFAULT_MAX_TURNS,
        toolChoice,
        signal,
    } = params;

    const preset = resolvePreset(presetName, presets, logger);

    // Build tool spec array for the wire protocol
    const toolSpecs: ToolSpec[] = tools.map(t => ({
        type: 'function' as const,
        function: {
            name: t.name,
            description: t.description,
            parameters: t.parameters,
        },
    }));

    // Fast lookup for execution
    const toolMap = new Map<string, AgentTool<TContext, TEffect>>(
        tools.map(t => [t.name, t]),
    );

    // Build initial wire message list
    const currentMessages: ChatMessage[] = [
        ...(system ? [{ role: 'system' as const, content: system }] : []),
        ...messages.map(m => ({ role: m.role as 'user' | 'assistant', content: m.content })),
    ];

    // Read cache: avoid redundant store reads within one loop run
    const readCache = new Map<string, string>();

    let responseText = '';
    const allEffects: TEffect[] = [];
    const allToolCalls: AgentToolRecord<TEffect>[] = [];
    let turn = 0;
    let consecutiveAllToolFailureTurns = 0;
    const aggregateUsage = {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
        estimatedCostUsd: 0,
    };

    // ── Tool context resolver ──────────────────────────────────────────────

    const resolveCtx = async (): Promise<TContext> => {
        if (getToolContext) return getToolContext();
        if (toolContext !== undefined) return toolContext;
        throw new Error(
            '[cortex] agent: either toolContext or getToolContext must be provided',
        );
    };

    // ── Safe tool executor ────────────────────────────────────────────────

    const executeToolSafely = async (
        tool: AgentTool<TContext, TEffect>,
        args: Record<string, unknown>,
        ctx: TContext,
    ): Promise<AgentToolResult<TEffect>> => {
        try {
            return await tool.execute(args, ctx);
        } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            logger.warn(`[cortex] agent: tool "${tool.name}" threw: ${msg}`);
            return { result: `Tool execution failed for "${tool.name}": ${msg}` };
        }
    };

    // ── Main loop ─────────────────────────────────────────────────────────

    logger.debug(
        `[cortex] agent: start — preset="${presetName}" tools=[${tools.map(t => t.name).join(', ')}]`,
    );

    while (true) {
        turn++;
        throwIfAborted(signal);

        if (turn > maxTurns) {
            if (!responseText) {
                responseText =
                    'I could not complete this request after multiple attempts. Please try again.';
            }
            logger.warn(`[cortex] agent: stopping — exceeded max turns (${maxTurns})`);
            break;
        }

        logger.debug(`[cortex] agent: turn ${turn} — sending to model`);

        const turnStartedAt = Date.now();
        const { result: response, capabilities: providerCapabilities } = await router.routeWithCapabilities(
            {
                preset,
                messages: currentMessages,
                tools: toolSpecs.length > 0 ? toolSpecs : undefined,
                // Only enforce toolChoice on the first turn — after the model has
                // called tools and we're feeding results back, let it decide freely
                // so it can choose to stop rather than loop forever.
                toolChoice: turn === 1 ? toolChoice : undefined,
                signal,
            },
            preferredProvider,
        );

        const turnDurationMs = Date.now() - turnStartedAt;

        // Accumulate + emit per-turn usage
        aggregateUsage.promptTokens += response.usage.prompt_tokens;
        aggregateUsage.completionTokens += response.usage.completion_tokens;
        aggregateUsage.totalTokens += response.usage.total_tokens;
        if (response.usage.cost !== undefined) {
            aggregateUsage.estimatedCostUsd += response.usage.cost;
        }
        if (onUsage) {
            onUsage({
                preset: presetName,
                model: response.model,
                promptTokens: response.usage.prompt_tokens,
                completionTokens: response.usage.completion_tokens,
                totalTokens: response.usage.total_tokens,
                costUsd: response.usage.cost,
                web: false,
                durationMs: turnDurationMs,
            });
        }

        if (response.content) {
            responseText += response.content;
        }

        logger.debug(`[cortex] agent: turn ${turn} — finish_reason=${response.finishReason}`);

        // ── Normal completion ──────────────────────────────────────────────
        if (
            response.finishReason === 'stop' ||
            response.finishReason === 'length' ||
            response.finishReason === 'content_filter'
        ) {
            break;
        }

        // ── Tool calls ────────────────────────────────────────────────────
        if (response.finishReason === 'tool_calls') {
            const toolCalls: ToolCallWire[] = response.toolCalls;

            if (toolCalls.length === 0) {
                logger.warn(
                    '[cortex] agent: finish_reason=tool_calls but response contains no tool_calls — ending turn',
                );
                break;
            }

            // Append assistant message with tool_calls to history
            currentMessages.push({
                role: 'assistant',
                content: response.content,
                tool_calls: toolCalls,
            });

            // ── Execute tools: read-only in parallel, writes serial ────────
            const rawResults: Array<{ tc: ToolCallWire; args: Record<string, unknown>; r: AgentToolResult<TEffect> }> = [];
            let idx = 0;

            while (idx < toolCalls.length) {
                throwIfAborted(signal);

                const tc = toolCalls[idx];
                const args = parseArgs(tc.function.arguments, tc.function.name, logger);
                const tool = toolMap.get(tc.function.name);

                if (!tool) {
                    rawResults.push({
                        tc,
                        args,
                        r: { result: `Unknown tool: ${tc.function.name}` },
                    });
                    idx++;
                    continue;
                }

                if (isToolReadOnly(tool, args)) {
                    // Collect contiguous read-only run
                    const readBatch: Array<{ tc: ToolCallWire; args: Record<string, unknown>; tool: AgentTool<TContext, TEffect> }> = [];
                    while (idx < toolCalls.length) {
                        const next = toolCalls[idx];
                        const nextArgs = parseArgs(next.function.arguments, next.function.name, logger);
                        const nextTool = toolMap.get(next.function.name);
                        if (!nextTool || !isToolReadOnly(nextTool, nextArgs)) break;
                        readBatch.push({ tc: next, args: nextArgs, tool: nextTool });
                        idx++;
                    }

                    const readResults = await Promise.all(
                        readBatch.map(async ({ tc: rtc, args: rargs, tool: rtool }) => {
                            const cacheKey = `${rtc.function.name}:${JSON.stringify(rargs)}`;
                            const cached = readCache.get(cacheKey);
                            if (cached !== undefined) {
                                return {
                                    tc: rtc,
                                    args: rargs,
                                    r: { result: cached } as AgentToolResult<TEffect>,
                                };
                            }
                            const ctx = await resolveCtx();
                            const r = await executeToolSafely(rtool, rargs, ctx);
                            readCache.set(cacheKey, r.result);
                            return { tc: rtc, args: rargs, r };
                        }),
                    );
                    rawResults.push(...readResults);
                    continue;
                }

                // Mutating tool: serial, with context refresh
                const ctx = await resolveCtx();
                const r = await executeToolSafely(tool, args, ctx);
                rawResults.push({ tc, args, r });
                idx++;
            }

            // ── Collect effects + build batch ──────────────────────────────
            const batch: AgentToolRecord<TEffect>[] = rawResults.map(({ tc, args, r }) => ({
                name: tc.function.name,
                args,
                result: r.result,
                effect: r.effect,
                effects: r.effects,
            }));

            for (const { r } of rawResults) {
                if (r.effect !== undefined) allEffects.push(r.effect);
                if (r.effects) allEffects.push(...r.effects);
            }
            allToolCalls.push(...batch);

            // ── Check consecutive all-failure ──────────────────────────────
            const allFailed =
                batch.length > 0 &&
                batch.every(b => b.result.startsWith('Tool execution failed for "'));
            if (allFailed) {
                consecutiveAllToolFailureTurns++;
            } else {
                consecutiveAllToolFailureTurns = 0;
            }

            // ── Append tool results as individual messages ─────────────────
            for (const { tc, r } of rawResults) {
                currentMessages.push({
                    role: 'tool',
                    content: r.result,
                    tool_call_id: tc.id,
                });
            }

            // ── onToolBatch hook ──────────────────────────────────────────
            if (onToolBatch) {
                await onToolBatch(batch);
            }

            // For providers that resolve tools internally (e.g. Apple Foundation
            // Models), the native layer already ran the tools and returned end_turn.
            // We applied JS-side effects above; do NOT send results back to the model.
            if (providerCapabilities.resolvesToolsInternally) {
                logger.debug('[cortex] agent: provider resolves tools internally — stopping after effect application');
                break;
            }

            if (consecutiveAllToolFailureTurns >= MAX_CONSECUTIVE_ALL_FAILURE_TURNS) {
                if (!responseText) {
                    const summary = batch.map(b => b.result).slice(0, 3).join(' | ');
                    responseText = `I could not complete this request due to repeated tool failures. ${summary}`;
                }
                logger.warn(
                    `[cortex] agent: stopping — ${MAX_CONSECUTIVE_ALL_FAILURE_TURNS} consecutive all-failure turns`,
                );
                break;
            }

            continue;
        }

        // Unknown finish reason — stop safely
        logger.warn(`[cortex] agent: unknown finish_reason "${response.finishReason}" — ending`);
        break;
    }

    logger.debug(
        `[cortex] agent: complete — ${turn} turn(s), ${aggregateUsage.totalTokens} total tokens`,
    );

    return {
        text: responseText.trim(),
        effects: allEffects,
        toolCalls: allToolCalls,
        usage: aggregateUsage,
    };
}
