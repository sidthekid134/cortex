import type { Logger } from './types.js';

export const noopLogger: Logger = {
    trace: () => {},
    debug: () => {},
    info: () => {},
    warn: () => {},
    error: () => {},
};

export const consoleLogger: Logger = {
    trace: (msg, ...args) => console.debug(`[cortex:trace] ${msg}`, ...args),
    debug: (msg, ...args) => console.debug(`[cortex:debug] ${msg}`, ...args),
    info: (msg, ...args) => console.info(`[cortex:info] ${msg}`, ...args),
    warn: (msg, ...args) => console.warn(`[cortex:warn] ${msg}`, ...args),
    error: (msg, ...args) => console.error(`[cortex:error] ${msg}`, ...args),
};
