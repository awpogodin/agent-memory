import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { convexTest } from "convex-test";
import schema from "./schema.js";
import { api } from "./_generated/api.js";

const modules = import.meta.glob("./**/*.ts");

const OPENAI_EMBEDDINGS = "https://api.openai.com/v1/embeddings";
const OPENROUTER_BASE = "https://openrouter.ai/api/v1";
const OPENROUTER_EMBEDDINGS = `${OPENROUTER_BASE}/embeddings`;
const OPENROUTER_MODEL = "openai/text-embedding-3-small";

function vector(dimensions = 1536): number[] {
  const v = new Array(dimensions).fill(0);
  v[0] = 1;
  return v;
}

type FetchCall = { url: string; body: any };

/**
 * Mock global.fetch. Embedding requests get a vector of `dimensions` length;
 * chat completion requests are answered by `chat` in call order.
 */
function mockFetch(opts: {
  dimensions?: number;
  embeddingStatus?: number;
  chat?: Array<() => unknown>;
} = {}) {
  const calls: FetchCall[] = [];
  let chatIndex = 0;
  const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
    const body = init?.body ? JSON.parse(init.body as string) : undefined;
    calls.push({ url, body });

    if (url.endsWith("/embeddings")) {
      if (opts.embeddingStatus && opts.embeddingStatus !== 200) {
        return new Response("upstream error", { status: opts.embeddingStatus });
      }
      return Response.json({
        data: [{ embedding: vector(opts.dimensions) }],
      });
    }

    if (url.endsWith("/chat/completions")) {
      const next = opts.chat?.[chatIndex++];
      const content = JSON.stringify(next ? next() : []);
      return Response.json({ choices: [{ message: { content } }] });
    }

    throw new Error(`Unexpected fetch: ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

async function createMemory(
  t: ReturnType<typeof convexTest>,
  overrides: { title?: string; content?: string } = {},
): Promise<string> {
  return await t.mutation(api.mutations.create, {
    projectId: "p1",
    scope: "project",
    title: overrides.title ?? "Use pnpm",
    content: overrides.content ?? "Always use pnpm for installs",
    memoryType: "instruction",
  });
}

const embeddingCalls = (calls: FetchCall[]) =>
  calls.filter((c) => c.url.endsWith("/embeddings"));

describe("embeddings", () => {
  let t: ReturnType<typeof convexTest>;

  beforeEach(() => {
    t = convexTest(schema, modules);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("generateEmbedding defaults to OpenAI and text-embedding-3-small", async () => {
    const { calls } = mockFetch();
    const memoryId = await createMemory(t);

    await t.action(api.actions.generateEmbedding, {
      memoryId,
      embeddingApiKey: "sk-test",
    });

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(OPENAI_EMBEDDINGS);
    expect(calls[0].body.model).toBe("text-embedding-3-small");

    const memory = await t.query(api.queries.get, { memoryId });
    expect(memory?.embeddingId).toBeDefined();
  });

  it("semanticSearch defaults to OpenAI and text-embedding-3-small", async () => {
    const { calls } = mockFetch();

    await t.action(api.actions.semanticSearch, {
      projectId: "p1",
      query: "pnpm",
      embeddingApiKey: "sk-test",
    });

    expect(calls).toHaveLength(1);
    expect(calls[0].url).toBe(OPENAI_EMBEDDINGS);
    expect(calls[0].body.model).toBe("text-embedding-3-small");
  });

  it("uses the same custom base URL and model for documents and queries", async () => {
    const { calls } = mockFetch();
    const memoryId = await createMemory(t);

    await t.action(api.actions.generateEmbedding, {
      memoryId,
      embeddingApiKey: "sk-or-test",
      model: OPENROUTER_MODEL,
      embeddingBaseUrl: OPENROUTER_BASE,
    });
    const results = await t.action(api.actions.semanticSearch, {
      projectId: "p1",
      query: "package manager",
      embeddingApiKey: "sk-or-test",
      model: OPENROUTER_MODEL,
      embeddingBaseUrl: OPENROUTER_BASE,
    });

    expect(calls.map((c) => c.url)).toEqual([
      OPENROUTER_EMBEDDINGS,
      OPENROUTER_EMBEDDINGS,
    ]);
    expect(calls.map((c) => c.body.model)).toEqual([
      OPENROUTER_MODEL,
      OPENROUTER_MODEL,
    ]);
    expect(results.map((m: any) => m._id)).toEqual([memoryId]);
  });

  it("handles a trailing slash in the base URL", async () => {
    const { calls } = mockFetch();
    const memoryId = await createMemory(t);

    await t.action(api.actions.generateEmbedding, {
      memoryId,
      embeddingApiKey: "sk-or-test",
      embeddingBaseUrl: `${OPENROUTER_BASE}/`,
    });
    await t.action(api.actions.semanticSearch, {
      projectId: "p1",
      query: "pnpm",
      embeddingApiKey: "sk-or-test",
      embeddingBaseUrl: `${OPENROUTER_BASE}/`,
    });

    expect(calls.map((c) => c.url)).toEqual([
      OPENROUTER_EMBEDDINGS,
      OPENROUTER_EMBEDDINGS,
    ]);
  });

  it("embedAll passes model and base URL through", async () => {
    const { calls } = mockFetch();
    await createMemory(t);
    await createMemory(t, { title: "Tabs", content: "Use tabs" });

    const result = await t.action(api.actions.embedAll, {
      projectId: "p1",
      embeddingApiKey: "sk-or-test",
      model: OPENROUTER_MODEL,
      embeddingBaseUrl: OPENROUTER_BASE,
    });

    expect(result).toEqual({ embedded: 2, skipped: 0 });
    expect(calls.every((c) => c.url === OPENROUTER_EMBEDDINGS)).toBe(true);
    expect(calls.every((c) => c.body.model === OPENROUTER_MODEL)).toBe(true);
  });

  it("generateEmbedding throws a clear error on a dimension mismatch", async () => {
    mockFetch({ dimensions: 768 });
    const memoryId = await createMemory(t);

    await expect(
      t.action(api.actions.generateEmbedding, {
        memoryId,
        embeddingApiKey: "sk-test",
        model: "nomic-embed-text",
      }),
    ).rejects.toThrow(
      'Embedding model "nomic-embed-text" returned 768 dimensions; the index requires 1536',
    );
  });

  it("semanticSearch throws on a dimension mismatch instead of falling back", async () => {
    mockFetch({ dimensions: 768 });
    await createMemory(t);

    await expect(
      t.action(api.actions.semanticSearch, {
        projectId: "p1",
        query: "pnpm",
        embeddingApiKey: "sk-test",
        model: "nomic-embed-text",
      }),
    ).rejects.toThrow("returned 768 dimensions; the index requires 1536");
  });

  it("semanticSearch falls back to full-text search on API errors", async () => {
    const { calls } = mockFetch({ embeddingStatus: 500 });
    const memoryId = await createMemory(t);

    const results = await t.action(api.actions.semanticSearch, {
      projectId: "p1",
      query: "pnpm",
      embeddingApiKey: "sk-test",
    });

    expect(calls).toHaveLength(1);
    expect(results.map((m: any) => m._id)).toEqual([memoryId]);
  });

  it("semanticSearch uses full-text search without an embedding key", async () => {
    const { fetchMock } = mockFetch();
    const memoryId = await createMemory(t);

    const results = await t.action(api.actions.semanticSearch, {
      projectId: "p1",
      query: "pnpm",
    });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(results.map((m: any) => m._id)).toEqual([memoryId]);
  });
});

describe("ingest", () => {
  let t: ReturnType<typeof convexTest>;

  beforeEach(() => {
    t = convexTest(schema, modules);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  const providerArgs = {
    llmApiKey: "sk-or-test",
    llmModel: "anthropic/claude-haiku-4.5",
    llmBaseUrl: OPENROUTER_BASE,
    embeddingApiKey: "sk-or-test",
    embeddingModel: OPENROUTER_MODEL,
    embeddingBaseUrl: OPENROUTER_BASE,
  };

  it("uses llmBaseUrl and llmModel for chat completions", async () => {
    const { calls } = mockFetch({
      chat: [() => ["Use pnpm"], () => []],
    });

    await t.action(api.actions.ingest, {
      projectId: "p1",
      content: "We use pnpm here.",
      ...providerArgs,
    });

    const chatCalls = calls.filter((c) => c.url.endsWith("/chat/completions"));
    expect(chatCalls.map((c) => c.url)).toEqual([
      `${OPENROUTER_BASE}/chat/completions`,
      `${OPENROUTER_BASE}/chat/completions`,
    ]);
    expect(chatCalls.every((c) => c.body.model === providerArgs.llmModel)).toBe(
      true,
    );
  });

  it("embeds ADD decisions with the configured model and base URL", async () => {
    const { calls } = mockFetch({
      chat: [
        () => ["Use pnpm"],
        () => [{ action: "ADD", content: "Use pnpm", title: "Package manager" }],
      ],
    });

    const result = await t.action(api.actions.ingest, {
      projectId: "p1",
      content: "We use pnpm here.",
      ...providerArgs,
    });

    expect(result.results.map((r: any) => r.event)).toEqual(["added"]);
    const embeds = embeddingCalls(calls);
    expect(embeds).toHaveLength(1);
    expect(embeds[0].url).toBe(OPENROUTER_EMBEDDINGS);
    expect(embeds[0].body.model).toBe(OPENROUTER_MODEL);

    const memory = await t.query(api.queries.get, {
      memoryId: result.results[0].memoryId,
    });
    expect(memory?.embeddingId).toBeDefined();
  });

  it("re-embeds UPDATE decisions with the configured model and base URL", async () => {
    const memoryId = await createMemory(t);
    const { calls } = mockFetch({
      chat: [
        () => ["Use pnpm 9"],
        () => [
          {
            action: "UPDATE",
            existingMemoryId: memoryId,
            content: "Always use pnpm 9 for installs",
            title: "Use pnpm",
          },
        ],
      ],
    });

    const result = await t.action(api.actions.ingest, {
      projectId: "p1",
      content: "Upgrade to pnpm 9.",
      ...providerArgs,
    });

    expect(result.results.map((r: any) => r.event)).toEqual(["updated"]);
    const embeds = embeddingCalls(calls);
    expect(embeds).toHaveLength(1);
    expect(embeds[0].url).toBe(OPENROUTER_EMBEDDINGS);
    expect(embeds[0].body.model).toBe(OPENROUTER_MODEL);
    expect(embeds[0].body.input).toContain("Always use pnpm 9 for installs");

    const memory = await t.query(api.queries.get, { memoryId });
    expect(memory?.embeddingId).toBeDefined();
  });

  it("keeps OpenAI defaults when no provider options are passed", async () => {
    const { calls } = mockFetch({
      chat: [
        () => ["Use pnpm"],
        () => [{ action: "ADD", content: "Use pnpm", title: "Package manager" }],
      ],
    });

    await t.action(api.actions.ingest, {
      projectId: "p1",
      content: "We use pnpm here.",
      llmApiKey: "sk-test",
      embeddingApiKey: "sk-test",
    });

    expect(calls.map((c) => [c.url, c.body.model])).toEqual([
      ["https://api.openai.com/v1/chat/completions", "gpt-4.1-nano"],
      ["https://api.openai.com/v1/chat/completions", "gpt-4.1-nano"],
      [OPENAI_EMBEDDINGS, "text-embedding-3-small"],
    ]);
  });
});
