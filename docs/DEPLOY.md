# Deploying

There are two very different things, and only one of them is meant to be public.

| | Who runs it | Data | Public? |
|---|---|---|---|
| **Your MemBridge** | you, on your computer (`uvx ... quickstart`, or Docker) | your own history | **Never expose it.** It has one user. |
| **The sandbox** | a host (Fly, Render, any Docker host) | made-up data only, rebuilt on every start | Yes: read-only, no login, rate-limited |

MemBridge is deliberately **not** a multi-user hosted service: the point is that your memory stays yours. The sandbox exists so anyone can see the product without installing it.

## The sandbox's guarantees (tested in `tests/test_sandbox.py`, and probed against the built container)
- It never opens a user data folder; it creates a temporary store of synthetic data at start-up and forgets it at shutdown.
- Visitors cannot write, import, export, back up, merge, delete or change settings (403); `health`, `clients` and the other endpoints that reveal server details are 404.
- The only POSTs allowed are the retrieval tester and unlocking the *private view* with the published PIN (`123456`: the data is synthetic).
- 90 requests per minute per client address (429 after that); strict security headers; a `Host` allow-list.
- No semantic-model download, no outbound calls.

## Run it locally
```bash
docker compose --profile sandbox up -d sandbox      # http://127.0.0.1:8080
# or without Docker:
uv run membridge sandbox                              # http://127.0.0.1:8080
```

## Deploy the sandbox
The host check (which blocks DNS rebinding) accepts only the hostname the platform announces (Render: `RENDER_EXTERNAL_HOSTNAME`, Fly: `<FLY_APP_NAME>.fly.dev`) or `MEMBRIDGE_PUBLIC_HOST` (comma-separated) for any other host.

**Fly.io**
```bash
fly auth login
fly launch --no-deploy --copy-config --name <your-app>   # uses fly.toml + Dockerfile.sandbox
fly deploy
```
**Render**: New > Blueprint > this repo (`render.yaml`). Nothing else to set.
**Any Docker host**: `docker build -f Dockerfile.sandbox -t membridge-sandbox .` then `docker run -p 80:8080 -e MEMBRIDGE_PUBLIC_HOST=demo.example.org membridge-sandbox` behind HTTPS.
Cold start builds the synthetic store in about 2 seconds; free tiers that sleep will take a moment on the first visit.

## Run your own (private) dashboard in Docker
`MEMBRIDGE_DASHBOARD_TOKEN=<long random string> docker compose --profile app up -d`, then open `http://127.0.0.1:8787/?token=<that string>`. It is published on `127.0.0.1` only; do not put it on the internet.

## Releasing
CI (`.github/workflows/ci.yml`) runs lint, mypy, the tests (with a real FalkorDB service), the security tests on Windows, checks that the committed dashboard bundle matches `web/`, and builds both images
and probes the sandbox container. After changing anything under `web/`, run `uv run python scripts/sync_web.py` and commit `src/membridge/_web`.
