# Analyst chatbot — "Ask the analyst"

A grounded Q&A assistant over the current stock universe, backed by the
**codex-api** `/v1/chat` service. It appears as the **Ask the analyst** tab in
the Streamlit UI (`render_chat`) and answers only from the model's latest
snapshot for every stock in the artifact — research only, never advice.

## How it works
- `portfolio_advisor.chat.build_system_prompt(snapshot)` builds a compact
  pipe-delimited table of **all** stocks
  (`ticker | sector | signal | score | ret21d% | conf | P/E | ROE | D/E | vol%`)
  and wraps it in a grounding system prompt (~8k tokens for 500 stocks).
- On the first turn the table is sent as the codex-api `system` prompt; follow-up
  turns reuse the returned `conversation_id` (server-side memory), so the context
  is sent only once per conversation.
- `StockChat` sends a normal `User-Agent` (codex-api is behind Cloudflare, which
  returns **1010** for the default python-urllib UA) and retries a transient
  `codex_failed` or `409 conversation_busy` once.

## Configure (Coolify env on the UI app)
```
CODEX_API_URL=https://codex-api.ahoum.com
CODEX_API_KEY=cxk_...      # mint per-consumer via codex-api POST /admin/keys
```
Without `CODEX_API_KEY` the tab shows a "not configured" notice and the rest of
the app is unaffected. The key is a per-consumer codex-api key — revoke it alone
if it leaks.
