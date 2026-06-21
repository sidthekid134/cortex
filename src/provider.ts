/**
 * Provider abstraction layer.
 *
 * A `Provider` wraps an underlying LLM runtime (OpenRouter, Apple Foundation
 * Models, a local model, etc.) behind a uniform interface. The agent loop talks
 * only to a Provider; switching runtimes requires no changes to orchestration code.
 *
 * Consumers register additional providers via `CortexInstance.registerProvider()`.
 * The router resolves which provider to use for each agent turn based on the
 * requested provider ID, capability requirements, and runtime availability.
 */

import type {
    ChatMessage,
    ToolSpec,
    ToolChoice,
    ChatUsage,
    ToolCallWire,
    ToolChatResult,
} from './openrouter.js';
import type { Preset, Logger } from './types.js';
import { OpenRouterClient } from './openrouter.js';
import { noopLogger } from './logger.js';

// ---------------------------------------------------------------------------
// Capability descriptor
// ---------------------------------------------------------------------------

export interface ProviderCapabilities {
    /** Whether the provider supports LLM tool calling (function calling). */
    tools: boolean;
    /** Whether the provider accepts image inputs. */
    vision: boolean;
    /** Whether the provider supports structured JSON output mode. */
    structuredOutput: boolean;
    /**
     * Set to `true` for runtimes (e.g. Apple Foundation Models) that resolve
     * tool calls internally and always return `finishReason: 'stop'` even when
     * tool logic was executed. The agent loop will apply tool effects but will
     * not send tool results back to the model or continue the turn loop.
     */
    resolvesToolsInternally?: boolean;
    /**
     * When set, only tools whose names appear in this list are forwarded to the
     * provider. The router will fall back to the cloud provider for agent runs
     * that require tools outside this list.
     */
    toolWhitelist?: string[];
}

// ---------------------------------------------------------------------------
// Provider chat params / result
// ---------------------------------------------------------------------------

export interface ProviderChatParams {
    preset: Preset;
    messages: ChatMessage[];
    tools?: ToolSpec[];
    toolChoice?: ToolChoice;
    signal?: AbortSignal;
}

export interface ProviderChatResult {
    content: string | null;
    toolCalls: ToolCallWire[];
    finishReason: 'stop' | 'tool_calls' | 'length' | 'content_filter';
    model: string;
    usage: ChatUsage;
}

// ---------------------------------------------------------------------------
// Provider interface
// ---------------------------------------------------------------------------

export interface Provider {
    /** Unique identifier used to request this provider in AgentParams.provider. */
    readonly id: string;
    readonly capabilities: ProviderCapabilities;
    /**
     * Optional availability check. Called by the router before routing a turn.
     * When this returns `false`, the router falls back to the cloud provider.
     */
    isAvailable?(): boolean | Promise<boolean>;
    chat(params: ProviderChatParams): Promise<ProviderChatResult>;
}

// ---------------------------------------------------------------------------
// OpenRouterProvider — wraps OpenRouterClient.chatWithTools()
// ---------------------------------------------------------------------------

export class OpenRouterProvider implements Provider {
    readonly id = 'openrouter';
    readonly capabilities: ProviderCapabilities = {
        tools: true,
        vision: true,
        structuredOutput: true,
    };

    constructor(private readonly client: OpenRouterClient) {}

    async chat(params: ProviderChatParams): Promise<ProviderChatResult> {
        const result: ToolChatResult = await this.client.chatWithTools(params.preset, params.messages, {
            tools: params.tools,
            toolChoice: params.toolChoice,
            signal: params.signal,
        });
        return result;
    }
}

// ---------------------------------------------------------------------------
// OnDeviceProvider — interface + no-op stub
// ---------------------------------------------------------------------------

/**
 * Interface that any on-device (local model) provider must implement.
 * Implementations are registered at runtime via `CortexInstance.registerProvider()`.
 *
 * The `resolvesToolsInternally` capability flag must be set to `true` for
 * runtimes (e.g. Apple Foundation Models) that resolve tool calls in the
 * native layer and return `finishReason: 'stop'` even when tool effects were
 * applied. The agent loop will execute the tool's JS side-effects but will
 * not send tool results back to the model or loop again.
 */
export interface OnDeviceProvider extends Provider {
    readonly capabilities: ProviderCapabilities & { resolvesToolsInternally: boolean };
}

/**
 * No-op on-device provider stub. Always reports itself as unavailable.
 * Register a real implementation via `CortexInstance.registerProvider()` once
 * the on-device runtime details (vision support, tool whitelist, perf
 * characteristics) are finalised.
 */
export const noOpOnDeviceProvider: OnDeviceProvider = {
    id: 'on-device',
    capabilities: {
        tools: false,
        vision: false,
        structuredOutput: false,
        resolvesToolsInternally: false,
    },
    isAvailable: () => false,
    async chat(): Promise<ProviderChatResult> {
        throw new Error('[cortex] on-device provider is not yet implemented');
    },
};

// ---------------------------------------------------------------------------
// ProviderRouter
// ---------------------------------------------------------------------------

/**
 * Routes agent turn requests to the appropriate provider based on:
 *   1. Preferred provider ID (from AgentParams.provider), if registered.
 *   2. Provider availability (isAvailable()).
 *   3. Capability requirements (tools, vision, tool whitelist).
 *
 * Falls back to the cloud (OpenRouter) provider on any mismatch.
 * Routing decisions are logged at debug level; errors still propagate to the
 * caller — the router never silently swallows failures.
 */
export class ProviderRouter {
    private readonly providers = new Map<string, Provider>();
    private readonly cloudProvider: Provider;
    private readonly logger: Logger;

    constructor(cloudProvider: Provider, logger: Logger = noopLogger) {
        this.cloudProvider = cloudProvider;
        this.logger = logger;
        this.providers.set(cloudProvider.id, cloudProvider);
    }

    registerProvider(provider: Provider): void {
        this.providers.set(provider.id, provider);
        this.logger.debug(`[cortex] provider registered: ${provider.id}`);
    }

    /**
     * Route a turn to the appropriate provider, returning both the result and
     * the capabilities of the provider that handled the request. Callers can
     * use the capabilities to adjust post-processing (e.g. resolvesToolsInternally).
     */
    async routeWithCapabilities(
        params: ProviderChatParams,
        preferredProviderId?: string,
    ): Promise<{ result: ProviderChatResult; capabilities: ProviderCapabilities }> {
        let provider = this.resolvePreferred(preferredProviderId);

        if (provider) {
            const available = provider.isAvailable ? await provider.isAvailable() : true;
            if (!available) {
                this.logger.info(
                    `[cortex] provider "${provider.id}" unavailable — routing to cloud`,
                );
                provider = null;
            }
        }

        if (provider && params.tools?.length) {
            if (!provider.capabilities.tools) {
                this.logger.warn(
                    `[cortex] provider "${provider.id}" does not support tools — routing to cloud`,
                );
                provider = null;
            } else if (provider.capabilities.toolWhitelist) {
                const required = params.tools.map(t => t.function.name);
                const unsupported = required.filter(
                    n => !provider!.capabilities.toolWhitelist!.includes(n),
                );
                if (unsupported.length > 0) {
                    this.logger.warn(
                        `[cortex] provider "${provider.id}" toolWhitelist excludes [${unsupported.join(', ')}] — routing to cloud`,
                    );
                    provider = null;
                }
            }
        }

        const target = provider ?? this.cloudProvider;
        if (target !== this.cloudProvider && preferredProviderId) {
            this.logger.debug(`[cortex] routing turn to "${target.id}"`);
        }
        const result = await target.chat(params);
        return { result, capabilities: target.capabilities };
    }

    async route(
        params: ProviderChatParams,
        preferredProviderId?: string,
    ): Promise<ProviderChatResult> {
        let provider = this.resolvePreferred(preferredProviderId);

        if (provider) {
            const available = provider.isAvailable ? await provider.isAvailable() : true;
            if (!available) {
                this.logger.info(
                    `[cortex] provider "${provider.id}" unavailable — routing to cloud`,
                );
                provider = null;
            }
        }

        if (provider && params.tools?.length) {
            if (!provider.capabilities.tools) {
                this.logger.warn(
                    `[cortex] provider "${provider.id}" does not support tools — routing to cloud`,
                );
                provider = null;
            } else if (provider.capabilities.toolWhitelist) {
                const required = params.tools.map(t => t.function.name);
                const unsupported = required.filter(
                    n => !provider!.capabilities.toolWhitelist!.includes(n),
                );
                if (unsupported.length > 0) {
                    this.logger.warn(
                        `[cortex] provider "${provider.id}" toolWhitelist excludes [${unsupported.join(', ')}] — routing to cloud`,
                    );
                    provider = null;
                }
            }
        }

        const target = provider ?? this.cloudProvider;
        if (target !== this.cloudProvider && preferredProviderId) {
            this.logger.debug(`[cortex] routing turn to "${target.id}"`);
        }
        return target.chat(params);
    }

    private resolvePreferred(id?: string): Provider | null {
        if (!id) return null;
        const p = this.providers.get(id);
        if (!p) {
            this.logger.warn(
                `[cortex] preferred provider "${id}" not registered — routing to cloud`,
            );
            return null;
        }
        return p;
    }
}
