// ---------------------------------------------------------------------------
// Main factory + instance type
// ---------------------------------------------------------------------------
export { createAIQueue } from './createAIQueue.js';
export type { CortexInstance } from './createAIQueue.js';

// ---------------------------------------------------------------------------
// Config + call parameter types
// ---------------------------------------------------------------------------
export type {
    CortexConfig,
    ConcurrencyConfig,
    PresetMap,
    Preset,
    Logger,
    LogLevel,
    StructuredParams,
    TextParams,
    ImageInput,
    UsageEvent,
    AITaskContext,
    AIOperationOptions,
    QueueStats,
    QueueEvent,
    QueueEventKind,
    QueueListener,
} from './types.js';

// ---------------------------------------------------------------------------
// Agent / tool-calling types
// ---------------------------------------------------------------------------
export type {
    AgentTool,
    AgentToolResult,
    AgentToolRecord,
    AgentMessage,
    AgentParams,
    AgentResult,
} from './types.js';

// ---------------------------------------------------------------------------
// Provider abstraction (for registering on-device or custom providers)
// ---------------------------------------------------------------------------
export type {
    Provider,
    ProviderCapabilities,
    ProviderChatParams,
    ProviderChatResult,
    OnDeviceProvider,
} from './provider.js';

export { noOpOnDeviceProvider } from './provider.js';

// ---------------------------------------------------------------------------
// Errors — import these to distinguish abort/cancel from real failures
// ---------------------------------------------------------------------------

/** Thrown on explicit cancellation (cancelGroup, cancelKey, destroy, or signal abort). */
export { AIOperationCancelledError, isAIAbortError } from './queue.js';

/**
 * Thrown when all structured() retry attempts are exhausted.
 * Check `.cause` for the last parse or Zod validation error.
 */
export { StructuredOutputError } from './structured.js';

/**
 * Thrown when the OpenRouter API returns a non-2xx status or an unexpected
 * payload. Check `.status` for the HTTP code and `.isRetryable` for whether
 * the request can be retried safely.
 */
export { OpenRouterError } from './openrouter.js';

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

/** Built-in preset catalog. Spread or override via `createAIQueue({ presets })`. */
export { DEFAULT_PRESETS } from './presets.js';

// ---------------------------------------------------------------------------
// Logging utilities
// ---------------------------------------------------------------------------

/**
 * Create a console-backed Logger filtered to a minimum level.
 * @example createLogger('warn') — only warnings and errors
 */
export { createLogger, noopLogger, consoleLogger } from './logger.js';
