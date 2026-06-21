import type { Logger, LogLevel } from './types.js';

// ---------------------------------------------------------------------------
// Level ordering
// ---------------------------------------------------------------------------

const LEVEL_VALUE: Record<LogLevel, number> = {
    trace: 10,
    debug: 20,
    info: 30,
    warn: 40,
    error: 50,
    silent: 100,
};

const noop = (): void => {};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Create a console-backed `Logger` that emits messages at or above the given
 * minimum level. All messages are prefixed with `[cortex]`.
 *
 * Pass `'silent'` to suppress all output — equivalent to `noopLogger`.
 *
 * @example
 * // Show only warnings and errors in production
 * const ai = createAIQueue({ apiKey, logger: createLogger('warn') });
 *
 * @example
 * // Show everything including trace-level internals during development
 * const ai = createAIQueue({ apiKey, logger: createLogger('trace') });
 */
export function createLogger(minLevel: LogLevel = 'debug'): Logger {
    const min = LEVEL_VALUE[minLevel];
    return {
        trace: min <= 10 ? (msg, ...args) => console.debug(`[cortex:trace] ${msg}`, ...args) : noop,
        debug: min <= 20 ? (msg, ...args) => console.debug(`[cortex] ${msg}`, ...args) : noop,
        info:  min <= 30 ? (msg, ...args) => console.info(`[cortex] ${msg}`, ...args) : noop,
        warn:  min <= 40 ? (msg, ...args) => console.warn(`[cortex:warn] ${msg}`, ...args) : noop,
        error: min <= 50 ? (msg, ...args) => console.error(`[cortex:error] ${msg}`, ...args) : noop,
    };
}

// ---------------------------------------------------------------------------
// Built-in loggers
// ---------------------------------------------------------------------------

/**
 * Silent logger — emits nothing. Used as the default when no logger is supplied
 * to `createAIQueue`. Swap this for `consoleLogger` during development.
 */
export const noopLogger: Logger = createLogger('silent');

/**
 * Console logger that emits all log levels (`debug` and above).
 * Convenient for development; use `createLogger('warn')` in production
 * if you want only warnings and errors.
 */
export const consoleLogger: Logger = createLogger('debug');
