import type { UsageEvent } from './types.js';
import type { ChatUsage } from './openrouter.js';
import type { AIOperationQueue } from './queue.js';

/**
 * Called after every successful OpenRouter call. Records usage into the queue's
 * aggregate totals and fires the app-supplied onUsage callback.
 */
export function recordCallUsage(
    queue: AIOperationQueue,
    event: Omit<UsageEvent, 'promptTokens' | 'completionTokens' | 'totalTokens' | 'costUsd'> & {
        usage: ChatUsage;
    },
    onUsage?: (e: UsageEvent) => void,
): void {
    const { prompt_tokens, completion_tokens, total_tokens, cost } = event.usage;

    queue.recordUsage(prompt_tokens, completion_tokens, cost);

    if (onUsage) {
        onUsage({
            preset: event.preset,
            model: event.model,
            promptTokens: prompt_tokens,
            completionTokens: completion_tokens,
            totalTokens: total_tokens,
            costUsd: cost,
            web: event.web,
            durationMs: event.durationMs,
        });
    }
}
