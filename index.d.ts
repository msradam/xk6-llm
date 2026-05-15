declare module 'k6/x/llm' {
  export interface SLOPredicate {
    /** TTFT threshold in milliseconds. 0 or omitted disables this SLO. */
    ttft_ms?: number;
    /** TPOT threshold in milliseconds. 0 or omitted disables this SLO. */
    tpot_ms?: number;
    /** End-to-end latency threshold in milliseconds. 0 or omitted disables this SLO. */
    e2el_ms?: number;
  }

  export interface ClientOptions {
    base_url?: string;
    api_key?: string;
    model?: string;
    timeout_ms?: number;
    ignore_eos?: boolean;
    /** Default SLO applied to every chat() call that does not override it. */
    slo?: SLOPredicate;
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
    /** Scalar (e2e - ttft) / (n - 1). 0 when not derivable (completion_tokens <= 1). */
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

  const _default: { Client: typeof Client };
  export default _default;
}
