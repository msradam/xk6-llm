declare module 'k6/x/llm' {
  export interface ClientOptions {
    base_url?: string;
    api_key?: string;
    model: string;
    timeout_ms?: number;
    ignore_eos?: boolean;
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
    [extra: string]: unknown;
  }

  export interface ChatResult {
    content: string;
    ttft_ms: number;
    /** Per-chunk inter-token latencies in milliseconds. Length = content_chunks - 1. */
    itl_ms: number[];
    duration_ms: number;
    prompt_tokens: number;
    completion_tokens: number;
    finish_reason: string;
  }

  export class Client {
    constructor(opts: ClientOptions);
    chat(req: ChatRequest): Promise<ChatResult>;
  }

  const _default: { Client: typeof Client };
  export default _default;
}
