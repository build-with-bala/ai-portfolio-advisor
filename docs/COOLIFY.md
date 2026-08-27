# Coolify handoff

## Single UI service

1. Push this repository to GitHub.
2. Create a Coolify application from the repository.
3. Select the repository Dockerfile and expose port `8501`.
4. Add a persistent volume or private artifact mount at `/app/data`.
5. Copy the Colab-generated `artifacts.pkl` into that mount.
6. Deploy and verify `/_stcore/health` through the container logs or the
   Coolify health check.
7. Add the domain after the health check is green.

Set the public FQDN to `https://fn.iimbg.com` after DNS points that hostname to
Coolify. The repository is deployment-ready for that hostname; creating the
Coolify application and DNS record is still an operator action.

The UI container is deliberately the public surface. It loads the artifact
read-only and never connects to a broker.

## Two-service UI + API

Use `docker-compose.yml` when the API should be available separately. Keep
the API behind Coolify's private network or an authenticated reverse proxy;
the example API currently provides analytical endpoints, not user accounts.

```text
public domain -> ui:8501
internal callers -> api:8000/health and /api/v1/*
shared read-only data -> /app/data/artifacts.pkl
```

Before production exposure, add authentication, request logging without
portfolio secrets, rate limiting, and a deployment-specific CORS allowlist.

For the Zerodha live adapter, put `KITE_API_KEY`, `KITE_ACCESS_TOKEN`, and
`KITE_INSTRUMENT_TOKENS` in Coolify's secret environment. The API secret is
used outside the running quote process to obtain or refresh the access token;
it must never be committed or returned by the app.

For the Upstox live adapter, set `LIVE_PROVIDER=upstox`,
`UPSTOX_ACCESS_TOKEN`, and `UPSTOX_INSTRUMENT_KEYS` in Coolify's secret
environment. `UPSTOX_INSTRUMENT_KEYS` is JSON mapping the artifact ticker to
an Upstox key, such as `{"RELIANCE.NS":"NSE_EQ|<ISIN>"}`. The access token
comes from Upstox OAuth and expires on the broker's schedule; an app key and
secret alone cannot start the feed. `UPSTOX_CLIENT_ID` and
`UPSTOX_REDIRECT_URI` may be set to show the OAuth connect link in the UI, but
the client secret must remain server-side and must never be committed.
If `UPSTOX_INSTRUMENT_KEYS` is omitted, the app resolves the artifact's `.NS`
symbols from Upstox's public daily NSE instrument master. The default live
subscription cap is 100 instruments; the research universe is not reduced by
that live-feed cap.
