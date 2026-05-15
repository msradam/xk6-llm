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

  export interface ChatMessage {
    role: 'system' | 'user' | 'assistant' | 'tool';
    content: string;
    name?: string;
  }

  export interface ChatRequest {
    messages: ChatMessage[];
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
    stop?: string | string[];
    seed?: number;

    /** Per-call SLO override. Falls back to ClientOptions.slo when absent. */
    slo?: SLOPredicate;
    /** Cache state for prefix-cache-aware aggregation. Emitted as a metric tag. */
    cache_state?: 'cold' | 'warm';
    /** Arbitrary request-scoped tags applied to all per-request samples. */
    tags?: Record<string, string>;

    [extra: string]: unknown;
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
  }

  export class Client {
    constructor(opts?: ClientOptions);
    chat(req: ChatRequest): Promise<ChatResult>;
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

  const _default: { Client: typeof Client; Dataset: typeof Dataset };
  export default _default;
}
