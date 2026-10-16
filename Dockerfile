# MemBridge application image (dashboard + HTTP API). FalkorDB runs as its own compose service.
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN pip install --no-cache-dir uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev
COPY --from=web /web/dist ./web/dist
# The CLI refuses non-loopback binds unless it knows it is inside a container; compose publishes the port on 127.0.0.1 only.
ENV MEMBRIDGE_IN_CONTAINER=1 MEMBRIDGE_DATA_DIR=/data MEMBRIDGE_FALKOR_HOST=falkordb MEMBRIDGE_FALKOR_PORT=6379
RUN useradd --create-home app && mkdir /data && chown app /data
USER app
VOLUME /data
EXPOSE 8787
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s CMD ["/app/.venv/bin/python", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8787/api/health', timeout=4)"]
CMD ["/app/.venv/bin/membridge", "serve", "--host", "0.0.0.0", "--port", "8787"]
