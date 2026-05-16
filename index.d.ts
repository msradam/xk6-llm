declare module 'k6/x/llm' {
  export interface SLOPredicate {
    /** TTFT threshold in milliseconds. 0 or omitted disables this SLO. */
    ttft_ms?: number;
    /** TPOT threshold in milliseconds. 0 or omitted disables this SLO. */
    tpot_ms?: number;
    /** End-to-end latency threshold in milliseconds. 0 or omitted disables this SLO. */
    e2el_ms?: number;
  }

  export interface CostModel {
    /** Hosted-API pricing in USD per million prompt tokens. */
    usd_per_million_input_tokens?: number;
    /** Hosted-API pricing in USD per million generated tokens. */
    usd_per_million_output_tokens?: number;
  }

  export interface EnergyModel {
    /** Joules per prompt token. Calibrate per (GPU, model). */
    j_per_input_token?: number;
    /** Joules per generated token. Calibrate per (GPU, model). */
    j_per_output_token?: number;
    /** Idle GPU power in watts. Under concurrent load divide by expected per-VU concurrency. */
    idle_w?: number;
  }

  export interface ClientOptions {
    base_url?: string;
    api_key?: string;
    model?: string;
    timeout_ms?: number;
    ignore_eos?: boolean;
    /** Extra HTTP headers sent on every request (gateways, custom auth, tracing). */
    headers?: Record<string, string>;
    /** Default SLO applied to every chat() call that does not override it. */
    slo?: SLOPredicate;
    /** When set, emits llm_energy_j and llm_energy_j_per_token per request. */
    energy?: EnergyModel;
    /** When set, emits llm_cost_usd per request. */
    cost?: CostModel;
  }

  export interface ToolCallMessage {
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }

  export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string | null;
    name?: string;
    /** Present on prior `assistant` turns that invoked tools. */
    tool_calls?: ToolCallMessage[];
    /** Required on `role: "tool"` messages; pairs the result with the call. */
    tool_call_id?: string;
  }

  export interface ToolDefinition {
    type: 'function';
    function: {
      name: string;
      description?: string;
      /** JSON Schema for the function arguments. */
      parameters: Record<string, unknown>;
    };
  }

  export interface ChatRequest {
    messages: ChatMessage[];
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop?: string | string[];
    seed?: number;
    /** OpenAI tool definitions. The model may respond with tool_calls. */
    tools?: ToolDefinition[];
    tool_choice?: 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };

    /** Per-call SLO override. Falls back to ClientOptions.slo when absent. */
    slo?: SLOPredicate;
    /** Cache state for prefix-cache-aware aggregation. Emitted as a metric tag. */
    cache_state?: 'cold' | 'warm';
    /** Arbitrary request-scoped tags applied to all per-request samples. */
    tags?: Record<string, string>;

    /** Cancel the stream after this many wall-clock milliseconds. The HTTP
     *  body is closed, propagating cancellation to the upstream server.
     *  Partial result resolves with `aborted: true`. */
    abort_after_ms?: number;
    /** Cancel the stream after this many content-bearing chunks. Partial
     *  result resolves with `aborted: true`. */
    abort_after_tokens?: number;

    [extra: string]: unknown;
  }

  export interface ChatToolCall {
    id: string;
    name: string;
    /** Raw JSON string emitted by the model. The script must JSON.parse it. */
    arguments: string;
  }

  export interface ChatResult {
    content: string;
    ttft_ms: number;
    /** Per-chunk inter-arrival in milliseconds. Length equals chunks - 1. */
    itl_ms: number[];
    /**
     * Scalar (duration_ms - ttft_ms) / (completion_tokens - 1). Returns 0 when
     * not derivable (completion_tokens <= 1, or no TTFT, or duration <= ttft).
     */
    tpot_ms: number;
    duration_ms: number;
    response_headers_ms: number;
    /** Count of content-bearing chunks. */
    chunks: number;
    prompt_tokens: number;
    completion_tokens: number;
    finish_reason: string;
    /** True when the stream was cut short by abort_after_ms or abort_after_tokens.
     *  All other fields reflect the partial state at the cutoff. */
    aborted: boolean;
    /** Tool invocations assembled from the stream. Empty when the model
     *  produced only content. Arguments are raw JSON strings. */
    tool_calls: ChatToolCall[];
  }

  export interface EmbedRequest {
    /** One or more strings to embed. */
    input: string | string[];
    /** Per-call model override. Falls back to ClientOptions.model. */
    model?: string;
    /** Arbitrary request-scoped tags applied to per-request samples. */
    tags?: Record<string, string>;
  }

  export interface EmbedResult {
    model: string;
    /** Embeddings in input order. */
    embeddings: number[][];
    prompt_tokens: number;
    duration_ms: number;
    /** Number of input strings supplied. */
    inputs: number;
  }

  export class Client {
    constructor(opts?: ClientOptions);
    chat(req: ChatRequest): Promise<ChatResult>;
    /** POST /v1/embeddings. Accepts a single string or a batch. */
    embed(req: EmbedRequest): Promise<EmbedResult>;
  }

  export interface SessionOptions {
    /** System prompt prepended to every conversation. Survives reset(). */
    system?: string;
    /** Stable session identifier. Auto-generated when omitted. */
    id?: string;
  }

  export interface SessionTokens {
    /**
     * Cumulative prompt tokens summed across every chat() in the session.
     * Grows quadratically with turn count because each call resends the full
     * history; matches what hosted APIs bill.
     */
    prompt: number;
    /** Cumulative completion tokens generated by the assistant. */
    completion: number;
    /** prompt + completion. */
    total: number;
  }

  /** Argument to Session.send(): the user content, or an object that pairs
   *  content with any chat() option (max_tokens, temperature, slo, tags). */
  export type SessionSendArg = string | (Omit<ChatRequest, 'messages'> & { content: string });

  /**
   * Multi-turn conversation wrapper. Each send() appends the user message,
   * dispatches a chat() call, and on resolution appends the assistant reply
   * to history. Every call is auto-tagged `session_id`, `turn`, and a
   * `cache_state` of `cold` (turn 1) or `warm` (later turns) unless the
   * caller supplies its own.
   */
  export class Session {
    constructor(client: Client, opts?: SessionOptions);
    send(arg: SessionSendArg): Promise<ChatResult>;
    /** Clear history (keeping the system prompt) and rewind turn counter.
     *  Token totals are preserved. */
    reset(): void;
    /** Session identifier used in tags. */
    id(): string;
    /** Number of completed turns. */
    turn(): number;
    /** Deep copy of the conversation history. */
    messages(): ChatMessage[];
    /** Cumulative token usage. See SessionTokens. */
    tokens(): SessionTokens;
  }

  export interface DatasetOptions {
    /** Path to a JSONL file. Each line: `{"messages": [...], "max_tokens"?: N, ...}`. */
    path: string;
    /** Seed for the shuffle permutation. Defaults to 42. */
    seed?: number;
    /** When true, the load order is a deterministic seeded permutation. */
    shuffle?: boolean;
  }

  /**
   * Replayable corpus of chat requests. Loaded once per process and shared
   * across VUs by absolute path; each instance carries its own cursor and
   * shuffle permutation.
   */
  export class Dataset {
    constructor(opts: DatasetOptions);
    /** Number of items in the dataset. */
    size(): number;
    /** Advance the internal cursor; returns the next request, wrapping at end. */
    next(): ChatRequest;
    /** Direct index lookup (modulo size, negative-safe). Does not advance the cursor. */
    at(i: number): ChatRequest;
    /** Rewind the internal cursor. */
    reset(): void;
  }

  const _default: { Client: typeof Client; Dataset: typeof Dataset; Session: typeof Session };
  export default _default;
}
