import { LLMOptions, CompletionOptions } from "../../index.js";
import { osModelsEditPrompt } from "../templates/edit.js";
import { streamSse } from "@continuedev/fetch";
import OpenAI from "./OpenAI.js";

const TOKEN_URL = "https://chat.qwen.ai/api/v1/oauth2/token";
const TOKEN_REFRESH_INTERVAL_MS = 3 * 60 * 60 * 1000; // 3 hours

class Qwen extends OpenAI {
  static providerName = "qwen";
  static defaultOptions: Partial<LLMOptions> = {
    apiBase: "https://chat.qwen.ai/api/v1/",
    model: "qwen3-coder-plus",
    promptTemplates: {
      edit: osModelsEditPrompt,
    },
    useLegacyCompletionsEndpoint: false,
  };

  maxStopWords: number | undefined = 16;

  // Internal token state
  private _accessToken: string = "";
  private _refreshToken: string = "";
  private _clientId: string = "";
  private _lastRefreshTime: number = 0;
  private _tokenInitialized: boolean = false;

  constructor(options: LLMOptions) {
    super(options);

    // User provides these in config.yaml:
    // apiKey       = access_token  (current OAuth token)
    // refreshToken = refresh_token
    // clientId     = client_id
    this._accessToken = options.apiKey ?? "";
    this._refreshToken = (options as any).refreshToken ?? "";
    this._clientId = (options as any).clientId ?? "";
    this._lastRefreshTime = Date.now();
    this._tokenInitialized = true;
  }

  // ─── Token Refresh Logic ──────────────────────────────────────────────────

  private async _refreshAccessToken(): Promise<void> {
    if (!this._refreshToken || !this._clientId) {
      throw new Error(
        "[Qwen] Cannot refresh token: refreshToken or clientId is missing in config."
      );
    }

    const body = new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: this._refreshToken,
      client_id: this._clientId,
    }).toString();

    const response = await fetch(TOKEN_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        Accept: "application/json",
      },
      body,
    });

    if (!response.ok) {
      throw new Error(
        `[Qwen] Token refresh failed: ${response.status} ${response.statusText}`
      );
    }

    const data = await response.json();

    if (!data.access_token) {
      throw new Error("[Qwen] Token refresh response missing access_token");
    }

    // Update tokens
    this._accessToken = data.access_token;
    if (data.refresh_token) {
      this._refreshToken = data.refresh_token;
    }
    this._lastRefreshTime = Date.now();

    console.log("[Qwen] Token refreshed successfully.");
  }

  private _isTokenExpired(): boolean {
    return Date.now() - this._lastRefreshTime >= TOKEN_REFRESH_INTERVAL_MS;
  }

  // Ensure token is fresh before every request
  private async _ensureFreshToken(): Promise<void> {
    if (this._isTokenExpired()) {
      await this._refreshAccessToken();
    }
  }

  // ─── Headers ──────────────────────────────────────────────────────────────

  protected _getHeaders() {
    return {
      "Content-Type": "application/json",
      Accept: "application/json",
      Authorization: `Bearer ${this._accessToken}`,
    };
  }

  // ─── Core: retry once on 401 by refreshing token ─────────────────────────

  private async _fetchWithRefresh(
    url: URL | string,
    init: RequestInit,
    signal: AbortSignal,
  ): Promise<Response> {
    // Ensure fresh before request
    await this._ensureFreshToken();

    const headersWithToken = {
      ...this._getHeaders(),
    };

    let response = await this.fetch(url, {
      ...init,
      headers: headersWithToken,
      signal,
    });

    // If 401, try refreshing once then retry
    if (response.status === 401) {
      console.warn("[Qwen] Got 401, attempting token refresh...");
      await this._refreshAccessToken();

      response = await this.fetch(url, {
        ...init,
        headers: this._getHeaders(),
        signal,
      });

      if (response.status === 401) {
        throw new Error(
          "[Qwen] Still getting 401 after token refresh. " +
          "Please check your refreshToken and clientId in config."
        );
      }
    }

    return response;
  }

  // ─── Stream Chat ──────────────────────────────────────────────────────────

  protected async *_streamChat(
    messages: any[],
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<any> {
    const body = this._convertArgs(options, messages);

    const response = await this._fetchWithRefresh(
      this._getEndpoint("chat/completions"),
      {
        method: "POST",
        body: JSON.stringify({ ...body }),
      },
      signal,
    );

    if (body.stream === false) {
      if (response.status === 499) return;
      const data = await response.json();
      yield data.choices[0].message;
      return;
    }

    for await (const value of streamSse(response)) {
      if (value.choices?.[0]?.delta) {
        yield {
          role: "assistant",
          content: value.choices[0].delta.content ?? "",
        };
      }
    }
  }

  // ─── FIM (autocomplete) ───────────────────────────────────────────────────

  supportsFim(): boolean {
    return true;
  }

  async *_streamFim(
    prefix: string,
    suffix: string,
    signal: AbortSignal,
    options: CompletionOptions,
  ): AsyncGenerator<string> {
    const endpoint = new URL("completions", this.apiBase);

    const response = await this._fetchWithRefresh(
      endpoint,
      {
        method: "POST",
        body: JSON.stringify({
          model: options.model,
          prompt: prefix,
          suffix,
          max_tokens: options.maxTokens,
          temperature: options.temperature,
          top_p: options.topP,
          frequency_penalty: options.frequencyPenalty,
          presence_penalty: options.presencePenalty,
          stop: options.stop,
          stream: true,
        }),
      },
      signal,
    );

    for await (const chunk of streamSse(response)) {
      yield chunk.choices[0].text;
    }
  }
}

export default Qwen;
