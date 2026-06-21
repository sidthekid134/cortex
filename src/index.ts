// Main factory
export { createAIQueue } from './createAIQueue.js';
export type { CortexInstance } from './createAIQueue.js';

// Config + call param types
export type {
    CortexConfig,
    ConcurrencyConfig,
    PresetMap,
    Preset,
    Logger,
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

// Queue error helpers (apps need these to distinguish abort vs real failures)
export { AIOperationCancelledError, isAIAbortError } from './queue.js';

// Default presets (apps can use these as a reference or spread them)
export { DEFAULT_PRESETS } from './presets.js';

// Loggers
export { noopLogger, consoleLogger } from './logger.js';
