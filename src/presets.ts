import type { PresetMap } from './types.js';

/**
 * Default preset catalog. Apps may override individual keys or replace the
 * entire map by passing `presets` to `createAIQueue`.
 *
 * Model strings follow the OpenRouter convention: "provider/model-name".
 */
export const DEFAULT_PRESETS: PresetMap = {
    /**
     * Fast, multimodal-capable model. Good for image description, food
     * recognition, and other latency-sensitive vision tasks.
     */
    'fast-vision': {
        model: 'google/gemini-flash-1.5',
        fallbacks: ['google/gemini-flash-1.5-8b', 'openai/gpt-4o-mini'],
        temperature: 0.2,
        maxTokens: 2048,
    },

    /**
     * Low-cost text-only model. Good for classification, short extraction,
     * and anything that doesn't need vision.
     */
    'cheap-text': {
        model: 'google/gemini-flash-1.5-8b',
        fallbacks: ['openai/gpt-4o-mini', 'meta-llama/llama-3.1-8b-instruct:free'],
        temperature: 0.1,
        maxTokens: 1024,
    },

    /**
     * Higher-capability model for grounded, web-assisted, or multi-step
     * reasoning tasks (e.g. menu matching, restaurant lookup).
     */
    reasoning: {
        model: 'google/gemini-pro-1.5',
        fallbacks: ['openai/gpt-4o', 'anthropic/claude-3.5-sonnet'],
        temperature: 0.3,
        maxTokens: 4096,
    },
};

/** Merge app-supplied preset overrides on top of the defaults. */
export function resolvePresets(overrides?: PresetMap): PresetMap {
    if (!overrides) return DEFAULT_PRESETS;
    return { ...DEFAULT_PRESETS, ...overrides };
}
