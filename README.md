# @sid/cortex

A self-contained, React Native / Expo-safe TypeScript library that pairs a UX-aware AI operation queue with a thin OpenRouter-backed structured-call layer.

**The app owns:** orchestration (what to enqueue and when), prompts, Zod schemas, and all UI state.  
**Cortex owns:** scheduling, priority, dedupe, group-cancellation, calling OpenRouter, JSON parsing/validation, presets, and cost telemetry.

Runtime dependency: `zod` (peer). No Vercel AI SDK, no provider SDKs, no Node-only APIs.

---

## Install

Add as a git dependency:

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

// Define your schema and prompt in the app
const FoodSchema = z.object({
  items: z.array(z.object({ name: z.string(), calories: z.number() })),
});

// Enqueue work — the app controls when and what to enqueue
const result = await ai.enqueue(
  'recognize-food',
  async (ctx) => {
    return ai.structured({
      preset: 'fast-vision',
      schema: FoodSchema,
      system: 'You are a food recognition assistant. Return JSON only.',
      prompt: 'What food items are in this image? Include estimated calories.',
      images: [{ base64: myBase64Image, mimeType: 'image/jpeg' }],
      signal: ctx.signal,  // always pass signal through for cancellation
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

### `ai.cancelGroup(group, reason?)`

Cancel all queued and running operations belonging to a group.

### `ai.cancelKey(dedupeKey, reason?)`

Cancel queued and running operations matching a dedupe key.

### `ai.hasPending(dedupeKey)`

Returns `true` if any operation with this dedupe key is queued or running.

### `ai.getStats()`

Returns current queue statistics including aggregate token usage and cost.

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
1. Converts the Zod schema to JSON Schema via `z.toJSONSchema`.
2. Posts to OpenRouter with `response_format: json_schema`.
3. Parses and validates the response with `schema.parse`.
4. On parse/validation failure, retries up to `maxRetries` (default: 1) with a corrective message.
5. Records usage via `onUsage`.

Returns `z.infer<typeof schema>`.

### `ai.text({ preset, system?, prompt, images?, web?, signal? })`

Makes a plain-text LLM call. Returns the raw string response. No schema validation — the app post-processes the result.

---

## Presets

### Default presets

| Name | Model | Best for |
|------|-------|---------|
| `fast-vision` | `google/gemini-flash-1.5` | Image description, food recognition, low-latency vision |
| `cheap-text` | `google/gemini-flash-1.5-8b` | Classification, short extraction, text-only tasks |
| `reasoning` | `google/gemini-pro-1.5` | Grounded search, multi-step reasoning, menu matching |

All presets include fallback chains across providers.

### Extending presets

```ts
const ai = createAIQueue({
  apiKey,
  presets: {
    'my-custom': { model: 'openai/gpt-4o', temperature: 0.5 },
    // Override a default:
    'fast-vision': { model: 'google/gemini-flash-2.0', fallbacks: [] },
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
import { isAIAbortError } from '@sid/cortex';

ai.enqueue('my-task', async (ctx) => {
  // ...
}).catch((error) => {
  if (isAIAbortError(error)) {
    // Normal: cancelled by group/key cancellation or setFocusedGroup change
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
