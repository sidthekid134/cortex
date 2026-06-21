import { describe, it, expect } from 'vitest';
import { DEFAULT_PRESETS, resolvePresets } from '../presets.js';

describe('DEFAULT_PRESETS', () => {
    it('has fast-vision preset', () => {
        expect(DEFAULT_PRESETS['fast-vision']).toBeDefined();
        expect(DEFAULT_PRESETS['fast-vision'].model).toBeTruthy();
    });

    it('has cheap-text preset', () => {
        expect(DEFAULT_PRESETS['cheap-text']).toBeDefined();
        expect(DEFAULT_PRESETS['cheap-text'].model).toBeTruthy();
    });

    it('has reasoning preset', () => {
        expect(DEFAULT_PRESETS['reasoning']).toBeDefined();
        expect(DEFAULT_PRESETS['reasoning'].model).toBeTruthy();
    });

    it('all presets have at least one fallback', () => {
        for (const [name, preset] of Object.entries(DEFAULT_PRESETS)) {
            expect(preset.fallbacks?.length, `${name} should have fallbacks`).toBeGreaterThan(0);
        }
    });

    it('all presets have temperature defined', () => {
        for (const [name, preset] of Object.entries(DEFAULT_PRESETS)) {
            expect(preset.temperature, `${name} should have temperature`).toBeDefined();
        }
    });

    it('all presets have maxTokens defined', () => {
        for (const [name, preset] of Object.entries(DEFAULT_PRESETS)) {
            expect(preset.maxTokens, `${name} should have maxTokens`).toBeDefined();
        }
    });
});

describe('resolvePresets', () => {
    it('returns DEFAULT_PRESETS when no overrides provided', () => {
        const result = resolvePresets();
        expect(result).toEqual(DEFAULT_PRESETS);
    });

    it('merges overrides on top of defaults', () => {
        const custom = { 'my-preset': { model: 'custom/model', temperature: 0.7 } };
        const result = resolvePresets(custom);
        expect(result['my-preset']).toEqual(custom['my-preset']);
        expect(result['fast-vision']).toEqual(DEFAULT_PRESETS['fast-vision']);
    });

    it('allows overriding an existing preset', () => {
        const override = { 'fast-vision': { model: 'overridden/model' } };
        const result = resolvePresets(override);
        expect(result['fast-vision'].model).toBe('overridden/model');
        // Other defaults still present
        expect(result['cheap-text']).toEqual(DEFAULT_PRESETS['cheap-text']);
    });

    it('does not mutate DEFAULT_PRESETS', () => {
        const original = { ...DEFAULT_PRESETS };
        resolvePresets({ 'fast-vision': { model: 'mutated' } });
        expect(DEFAULT_PRESETS['fast-vision'].model).toBe(original['fast-vision'].model);
    });
});
