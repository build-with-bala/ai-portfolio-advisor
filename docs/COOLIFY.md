# Coolify handoff

## Single UI service

1. Push this repository to GitHub.
2. Create a Coolify application from the repository.
3. Select the repository Dockerfile and expose port `8501`.
4. Add a persistent volume or private artifact mount at `/app/data`.
5. Copy the Colab-generated `artifacts.pkl` into that mount.
6. Deploy and verify `/ _stcore/health` through the container logs or the
   Coolify health check. The actual path has no space: `/_stcore/health`.
7. Add the domain after the health check is green.

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
