# Live-market microservice

`live_service.py` splits the live NSE feed out of the Streamlit UI into its own
deployable process. It owns **one** upstream broker connection and the OAuth
token lifecycle; the UI (and anything else) consumes quotes over HTTP/SSE.

## Why

The feed used to run in-process inside Streamlit (`cached_live_service`), so
every UI session/replica opened its **own** broker WebSocket — brokers cap
concurrent connections, and the access token was managed per-process. The
microservice fixes all three:

- **one** broker WebSocket for the whole deployment,
- **central** OAuth token authority — the Upstox client secret lives only here,
- independent **restart / scale** from the research UI.

The daily research model is **not** loaded in the service; it is purely the live
quote + token authority. The UI keeps the model and blends the intraday overlay.

## Topology

```
Upstox V3 / Zerodha Kite WS
        │  (single connection)
        ▼
  live-service  (FastAPI, :8900)      ── HTTP / SSE ──▶  Streamlit UI
  LIVE_PROVIDER=upstox                                   LIVE_PROVIDER=remote
  /live/quotes  /live/stream                             LIVE_SERVICE_URL=http://live:8900
  /live/status  /live/tape                               → RemotePollingFeed mirrors quotes
  /live/oauth/upstox/exchange                            → existing blend logic unchanged
  /live/token
```

The UI becomes just another provider: `LIVE_PROVIDER=remote` makes
`LiveMarketService` use `RemotePollingFeed`, which polls the service's
`/live/quotes` and mirrors it into the local store — so every downstream
analysis path keeps working with no other change.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + provider + quote count |
| GET | `/live/status` | provider, NSE session, last tick, error |
| GET | `/live/quotes` | full quote snapshot |
| GET | `/live/quotes/{ticker}` | one quote |
| GET | `/live/tape` | in-process quote history (for charts) |
| GET | `/live/stream?interval=3` | Server-Sent Events snapshot stream |
| GET | `/live/oauth/upstox/url` | Upstox login-dialog URL (needs client id + redirect) |
| POST | `/live/oauth/upstox/exchange` `{code}` | exchange OAuth code → token (secret server-side), (re)start feed |
| POST | `/live/token` `{access_token, provider}` | set a broker token directly, restart feed |

## Run locally

```bash
docker compose up --build          # ui:8501, api:8000, live:8900 (paper mode)
# go live:
LIVE_PROVIDER=upstox docker compose up --build   # then set the Upstox env below
```

## Deploy on Coolify (BWB VPS)

Add a **second application** from this repo using `Dockerfile.live`, expose
`8900`, mount the shared read-only artifact volume at `/app/data`, and keep it on
the **private** network. On the existing UI app, set `LIVE_PROVIDER=remote` and
`LIVE_SERVICE_URL=http://<live-service-internal>:8900`.

Broker secrets go in the live-service's **secret** env only (never in git):

```
LIVE_PROVIDER=upstox
UPSTOX_CLIENT_ID=...
UPSTOX_CLIENT_SECRET=...          # used only for the token exchange, server-side
UPSTOX_REDIRECT_URI=https://fn.iimbg.com
UPSTOX_ACCESS_TOKEN=...           # or POST /live/oauth/upstox/exchange after login
```

Upstox access tokens expire on the broker's daily schedule; re-run the OAuth
exchange (or set a fresh token) each session. `UPSTOX_INSTRUMENT_KEYS` may be
omitted — the service resolves the artifact's `.NS` symbols from Upstox's public
NSE instrument master, and the Nifty-500 constituent file already carries every
ISIN for an explicit mapping if preferred.
