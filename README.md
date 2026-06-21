# @sid/cortex

A UX-aware LLM job queue + OpenRouter integration layer for React Native / Expo apps.

Most apps fire LLM calls directly and end up managing a tangle of race conditions, duplicate requests, stale results, and no visibility into cost. Cortex puts a scheduler between your app and the model so you get **priority, deduplication, group cancellation, structured output with retries, and per-call cost telemetry**—without writing any of that infrastructure yourself.

**The app owns:** orchestration (what to enqueue and when), prompts, Zod schemas, and all UI state.  
**Cortex owns:** scheduling, priority, dedupe, group-cancellation, calling OpenRouter, JSON parsing/validation, presets, and cost telemetry.

Runtime dependency: `zod` (peer). No Vercel AI SDK, no provider SDKs, no Node-only APIs — just `fetch` and `AbortSignal`.

---

## The problem it solves

A typical feed-based app triggers an LLM call for each item a user scrolls past. Without a scheduler:

```
User scrolls past items A → B → C → D

  A ──────────────────────────────────────► response (stale, user left)
    B ─────────────────────────────────────► response (stale)
      C ──────────────────────────────────► response (stale)
        D ──────────────────────────────────► response ✓

  5 inflight requests. All run at equal priority.
  A/B/C waste tokens and cost money. D is no faster.
```

With Cortex:

```
User scrolls past items A → B → C → D
setFocusedGroup(D) on navigation

  A ──► cancelled (group cancelled on navigate)
  B ──► cancelled
  C ──► cancelled
  D ──────────────────────────────────────► response ✓

  1 inflight request. D runs immediately in a burst slot.
  Zero wasted tokens.
```

---

## How it works

```
┌─────────────────────────────────────────────────────────────┐
│                      Client App                             │
│  prompts · schemas · when to enqueue · UI state · tools     │
└──────────────────┬──────────────────────────────────────────┘
                   │  ai.enqueue(label, task, { group, priority, dedupeKey })
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    Cortex (AIOperationQueue)                 │
│                                                             │
│   Priority queue                                            │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  urgent (priority ≥ 90)  ──► burst slots (up to 5)  │  │
│   │  focused group           ──► bumped to front         │  │
│   │  background (default)    ──► base slots (up to 3)    │  │
│   │  slow ops                ──► slow slots (up to 2)    │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                             │
│   Deduplication                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  same dedupeKey pending  → bump priority, share       │  │
│   │  same dedupeKey running  → attach (or restart)       │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                             │
│   Cancellation                                              │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  cancelGroup(id)   → abort all ops in group          │  │
│   │  cancelKey(key)    → abort by dedupe key             │  │
│   │  isStillValid()    → skip before run                 │  │
│   └──────────────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────────────────┘
                   │  structured() / text() / agent()
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                OpenRouter (/chat/completions)                │
│         Zod → JSON Schema · validate · retry · telemetry    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key capabilities

| Capability | How it works |
|---|---|
| **Priority scheduling** | Higher priority runs first; FIFO within same level |
| **Focus-aware bursting** | `setFocusedGroup()` bumps that group to front + opens burst concurrency slots |
| **Deduplication** | Same `dedupeKey` pending → bump + share promise; running → attach or `restart` |
| **Group cancellation** | `cancelGroup(id)` aborts all queued and running ops in a group |
| **Structured output** | Zod schema → JSON Schema → parse/validate → retry with corrective prompt on failure |
| **Tool-calling agent** | Multi-turn loop; read-only tools run parallel, mutating tools serial |
| **Cost telemetry** | Per-call `UsageEvent` (tokens, cost, duration) + session aggregate via `getStats()` |
| **Provider abstraction** | OpenRouter default; plug in on-device models via `registerProvider()` |

---

## Install

```json
"dependencies": {
  "@sid/cortex": "git+https://github.com/sidmoparthi/cortex.git",
  "zod": "^4.0.0"
}
```

---

## Quick start

```ts
import { createAIQueue, isAIAbortError } from '@sid/cortex';
import { z } from 'zod';

const ai = createAIQueue({
  apiKey: process.env.OPENROUTER_API_KEY!,
  headers: { referer: 'https://myapp.com', title: 'MyApp' },
  onUsage: (e) => console.log(`${e.preset} → ${e.model} | $${e.costUsd} | ${e.durationMs}ms`),
});

const FoodSchema = z.object({
  items: z.array(z.object({ name: z.string(), calories: z.number() })),
});

const result = await ai.enqueue(
  'recognize-food',
  async (ctx) => {
    return ai.structured({
      preset: 'fast-vision',
      schema: FoodSchema,
      system: 'You are a food recognition assistant. Return JSON only.',
      prompt: 'What food items are in this image? Include estimated calories.',
      images: [{ base64: myBase64Image, mimeType: 'image/jpeg' }],
      signal: ctx.signal,
    });
  },
  {
    dedupeKey: `${entryId}:recognizeFood`,
    group: entryId,
    priority: 50,
    isStillValid: () => entryStillExists(entryId),
  },
);
```

---

## `createAIQueue(config)`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `apiKey` | `string` | required | OpenRouter API key |
| `baseUrl` | `string` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `presets` | `PresetMap` | merged over defaults | Override or extend the default preset catalog |
| `concurrency.base` | `number` | `3` | Max concurrent background operations |
| `concurrency.burst` | `number` | `2` | Extra slots for urgent / focused work |
| `concurrency.slowOp` | `number` | `base - 1` | Max concurrent slow operations |
| `interactiveThreshold` | `number` | `90` | Priority at/above which work is considered urgent |
| `headers.referer` | `string` | — | HTTP-Referer sent to OpenRouter |
| `headers.title` | `string` | — | X-Title sent to OpenRouter |
| `logger` | `Logger` | no-op | Injectable logger (pass `consoleLogger` for dev output) |
| `onUsage` | `(e: UsageEvent) => void` | — | Called after every completed LLM call |

Returns a `CortexInstance`.

---

## Queue API

### `ai.enqueue(label, task, options?)`

Add a task to the queue. Returns a promise that resolves/rejects when the task completes.

| Option | Type | Description |
|--------|------|-------------|
| `dedupeKey` | `string` | A newer enqueue with the same key supersedes the pending one |
| `group` | `string` | Cancellation group (e.g. an entity id). All ops in a group cancel together |
| `priority` | `number` | Higher = runs sooner. Default: `0` |
| `isStillValid` | `() => boolean` | Checked right before the task runs; returns false to skip it |
| `restart` | `boolean` | If the same dedupeKey is already running, abort and restart it |
| `slowOp` | `boolean` | Mark as a slow/long-running call; capped separately from fast calls |

### `ai.setFocusedGroup(group | null)`

Mark the group the user is actively viewing. Its queued operations are sorted to the front and may run in burst concurrency slots without cancelling already-running work.

```
Before setFocusedGroup('item-3'):        After setFocusedGroup('item-3'):

  Queue (FIFO):                            Queue (priority-sorted):
  1. item-1:recognize   [p=50]             1. item-3:recognize   [p=50, focused ↑]
  2. item-2:recognize   [p=50]             2. item-3:details     [p=30, focused ↑]
  3. item-3:recognize   [p=50]             3. item-1:recognize   [p=50]
  4. item-3:details     [p=30]             4. item-2:recognize   [p=50]

  Running: item-1:recognize (keeps running — no disruption)
  New burst slot opens for item-3 work immediately
```

### `ai.cancelGroup(group, reason?)`

Cancel all queued and running operations belonging to a group.

### `ai.cancelKey(dedupeKey, reason?)`

Cancel queued and running operations matching a dedupe key.

### `ai.hasPending(dedupeKey)`

Returns `true` if any operation with this dedupe key is queued or running.

### `ai.getStats()`

Returns current queue statistics including aggregate token usage and cost.

```ts
const stats = ai.getStats();
// {
//   runningOperations: 2,
//   pendingOperations: 5,
//   completedOperations: 14,
//   failedOperations: 0,
//   skippedOperations: 3,
//   cancelledOperations: 6,
//   usage: {
//     totalTokens: 48200,
//     promptTokens: 41000,
//     completionTokens: 7200,
//     estimatedCostUsd: 0.0097,
//   }
// }
```

### `ai.subscribe(listener)`

Subscribe to queue lifecycle events (`queued`, `bumped`, `started`, `completed`, `failed`, `cancelled`, `skipped`). Returns an unsubscribe function. Use this to mirror queue state into your own store without polling.

```ts
const unsub = ai.subscribe((event) => {
  console.log(event.kind, event.label, event.stats.runningOperations);
  myStore.setState({ queueStats: event.stats });
});
// later:
unsub();
```

---

## LLM call API

Both helpers are designed to be called **inside** a queued task so `ctx.signal` flows through.

### `ai.structured({ preset, schema, system?, prompt, images?, web?, signal?, maxRetries? })`

Makes a structured LLM call. Flow:

```
  prompt + Zod schema
        │
        ▼
  z.toJSONSchema(schema)
        │
        ▼
  POST /chat/completions
  response_format: json_schema
        │
        ▼
  JSON.parse + schema.parse
        │
    ┌───┴───┐
  valid   invalid
    │         │
    ▼         ▼ (up to maxRetries, default 1)
  return   retry with corrective message
             │
          ┌──┴──┐
        valid  invalid
          │       │
          ▼       ▼
        return  throw StructuredOutputError
```

Returns `z.infer<typeof schema>`.

### `ai.text({ preset, system?, prompt, images?, web?, signal? })`

Makes a plain-text LLM call. Returns the raw string response.

### `ai.agent({ preset, tools, system?, prompt, signal?, maxTurns? })`

Multi-turn tool-calling loop (default max 12 turns). Read-only tools run in parallel per turn; mutating tools run serially. Pass a `onToolBatch` hook to react to each batch of tool results.

> Agent inner turns bypass the queue intentionally to avoid self-deadlock on concurrency limits.

---

## Presets

### Default presets

| Name | Model | Best for |
|------|-------|---------|
| `fast-vision` | `google/gemini-2.5-flash` | Image description, food recognition, low-latency vision |
| `cheap-text` | `google/gemini-2.5-flash-lite` | Classification, short extraction, text-only tasks |
| `reasoning` | `google/gemini-2.5-pro` | Grounded search, multi-step reasoning, web lookups |

All presets include fallback chains across providers.

### Extending presets

```ts
const ai = createAIQueue({
  apiKey,
  presets: {
    'my-custom': { model: 'openai/gpt-4o', temperature: 0.5 },
    // Override a default:
    'fast-vision': { model: 'google/gemini-2.5-flash', fallbacks: [] },
  },
});
```

---

## Web access

Pass `web: true` to any call that may need internet access (e.g. looking up a restaurant menu, finding a product page):

```ts
const result = await ai.structured({
  preset: 'reasoning',
  schema: MenuSchema,
  prompt: `Find the menu for ${restaurantName} and extract all items with prices.`,
  web: true,
  signal: ctx.signal,
});
```

Cortex maps this to OpenRouter's web plugin (`plugins: [{ id: 'web' }]`). The model decides when to search.

---

## Error handling

```ts
import { isAIAbortError, StructuredOutputError, OpenRouterError } from '@sid/cortex';

ai.enqueue('my-task', async (ctx) => {
  // ...
}).catch((error) => {
  if (isAIAbortError(error)) {
    // Normal: cancelled by group/key cancellation or setFocusedGroup change
    return;
  }
  if (error instanceof StructuredOutputError) {
    // Exhausted retries — error.cause has the last Zod validation error
    console.error('Schema validation failed after retries:', error.cause);
    return;
  }
  if (error instanceof OpenRouterError) {
    // HTTP error — error.status, error.isRetryable (true for 429 / 5xx)
    console.error(`OpenRouter ${error.status}:`, error.message);
    return;
  }
  console.error('Task failed:', error);
});
```

---

## Usage telemetry

```ts
const ai = createAIQueue({
  apiKey,
  onUsage: (e) => {
    analytics.track('llm_call', {
      preset: e.preset,
      model: e.model,
      promptTokens: e.promptTokens,
      completionTokens: e.completionTokens,
      costUsd: e.costUsd,
      durationMs: e.durationMs,
    });
  },
});

// Aggregate totals are also in getStats():
const { usage } = ai.getStats();
console.log(`Session: ${usage.totalTokens} tokens, ~$${usage.estimatedCostUsd.toFixed(4)}`);
```

---

## Logging

```ts
import { createAIQueue, consoleLogger } from '@sid/cortex';

const ai = createAIQueue({
  apiKey,
  logger: consoleLogger, // prints trace/debug/info/warn/error to console
});
```

Bring your own logger by implementing the `Logger` interface.

---

## Concurrency reference

```
Default concurrency limits:

  ┌──────────────────────────────────────────────────────┐
  │                   Max 5 total slots                  │
  │                                                      │
  │  [ base slot ][ base slot ][ base slot ]             │  ← 3 base slots (all ops)
  │  [ burst slot ][ burst slot ]                        │  ← +2 for urgent / focused group
  │                                                      │
  │  slow ops: max 2 of the 3 base slots                 │
  │  (so at least 1 slot always free for fast ops)       │
  └──────────────────────────────────────────────────────┘

Priority thresholds:

  0        50        90       100
  │────────│─────────│────────│
  bg      normal   urgent   max
                     ↑
              interactiveThreshold
              (opens burst slots)
```
