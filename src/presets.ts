import type { PresetMap } from './types.js';

/**
 * Default preset catalog. Apps may override individual keys or add new ones by
 * passing `presets` to `createAIQueue`. To replace all defaults, pass only your
 * own keys; to extend them, include just the keys you want to change.
 *
 * Model strings follow the OpenRouter convention: `"provider/model-name"`.
 * @see https://openrouter.ai/models
 */
export const DEFAULT_PRESETS: PresetMap = {
    /**
     * Fast, multimodal-capable model. Optimised for latency-sensitive vision
     * tasks — image description, food recognition, receipt parsing, etc.
     * Falls back to a smaller variant then GPT-4o-mini on error.
     */
    'fast-vision': {
        model: 'google/gemini-2.0-flash',
        fallbacks: ['google/gemini-2.0-flash-lite', 'openai/gpt-4o-mini'],
        temperature: 0.2,
        maxTokens: 2048,
    },

    /**
     * Low-cost text-only model. Ideal for classification, short extraction,
     * and tasks that do not require vision or heavy reasoning.
     * Falls back to GPT-4o-mini then a free Llama variant.
     */
    'cheap-text': {
        model: 'google/gemini-2.0-flash-lite',
        fallbacks: ['openai/gpt-4o-mini', 'meta-llama/llama-3.1-8b-instruct:free'],
        temperature: 0.1,
        maxTokens: 1024,
    },

    /**
     * Higher-capability model for complex, multi-step, or web-assisted reasoning
     * tasks (e.g. menu matching, restaurant lookup, nutritional inference).
     * Falls back to GPT-4o then Claude Sonnet on error.
     */
    reasoning: {
        model: 'google/gemini-2.5-pro',
        fallbacks: ['openai/gpt-4o', 'anthropic/claude-sonnet-4-5'],
        temperature: 0.3,
        maxTokens: 4096,
    },
};

/**
 * Merge app-supplied preset overrides on top of the defaults.
 * Keys present in `overrides` replace the corresponding default; all other
 * defaults are preserved.
 */
export function resolvePresets(overrides?: PresetMap): PresetMap {
    if (!overrides) return DEFAULT_PRESETS;
    return { ...DEFAULT_PRESETS, ...overrides };
}
