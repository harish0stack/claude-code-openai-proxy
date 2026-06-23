# claude-code-openai-proxy

A 24/7 GitHub Actions proxy mapping Claude Code & Anthropic API requests to any OpenAI-compatible endpoint via ngrok.

A fast, lightweight proxy that sits between **Claude Code** (or any Anthropic SDK client) and **any OpenAI-compatible LLM endpoint** — OpenAI, NVIDIA NIM, Together AI, Groq, Azure OpenAI, OpenRouter, Ollama, and more.

```text
Claude Code  ──►  this proxy (translates format)  ──►  any OpenAI-compatible endpoint
             Anthropic /v1/messages              OpenAI /v1/chat/completions

Built with Python + `uv` + FastAPI + httpx (async, connection-pooled). Runs locally or on GitHub Actions (free minutes via GitHub Student Pack).

---

## Features

- ✅ Streaming SSE (real-time tokens)
- ✅ Tool/function calling (full round-trip)
- ✅ Image inputs (base64)
- ✅ System prompts
- ✅ Model mapping (Claude model names → provider model names)
- ✅ Extended thinking → reasoning model routing
- ✅ API key passthrough (one proxy, many users)
- ✅ Any OpenAI-compatible backend (one env-var change)
- ✅ `/health` endpoint for monitoring
- ✅ GitHub Actions workflow included

---

## Quick start (local)

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install deps
git clone https://github.com/YOU/anthropic-openai-proxy
cd anthropic-openai-proxy
uv sync

# 3. Configure
cp .env.example .env
# Edit .env — set UPSTREAM_BASE_URL and UPSTREAM_API_KEY

# 4. Run
uv run uvicorn src.proxy:app --host 0.0.0.0 --port 8082

# 5. Point Claude Code at it
export ANTHROPIC_BASE_URL=http://localhost:8082
export ANTHROPIC_API_KEY=dummy
claude
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_BASE_URL` | `https://api.openai.com` | Base URL of your OpenAI-compatible endpoint (no `/v1` suffix) |
| `UPSTREAM_API_KEY` | *(required)* | API key for the upstream provider |
| `UPSTREAM_API_KEY_PASSTHROUGH` | `false` | If `true`, use the `x-api-key` from each incoming request |
| `BIG_MODEL` | `gpt-4o` | Upstream model for opus/sonnet requests |
| `SMALL_MODEL` | `gpt-4o-mini` | Upstream model for haiku requests |
| `REASONING_MODEL` | *(empty)* | If set, requests with extended thinking go here (e.g. `o3`) |
| `MODEL_MAP` | *(empty)* | Explicit overrides: `claude-opus-4-6=gpt-4o;claude-haiku-4-5=gpt-4o-mini` |
| `PORT` | `8082` | Port the proxy listens on |

---

## Provider examples

### OpenAI
```env
UPSTREAM_BASE_URL=https://api.openai.com
UPSTREAM_API_KEY=sk-...
BIG_MODEL=gpt-4o
SMALL_MODEL=gpt-4o-mini
REASONING_MODEL=o3
```

### NVIDIA NIM
```env
UPSTREAM_BASE_URL=https://integrate.api.nvidia.com
UPSTREAM_API_KEY=nvapi-...
BIG_MODEL=nvidia/llama-3.1-nemotron-ultra-253b-v1
SMALL_MODEL=nvidia/llama-3.3-70b-instruct
```

### Together AI
```env
UPSTREAM_BASE_URL=https://api.together.xyz
UPSTREAM_API_KEY=...
BIG_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo
SMALL_MODEL=meta-llama/Llama-3.2-3B-Instruct-Turbo
```

### Groq
```env
UPSTREAM_BASE_URL=https://api.groq.com/openai
UPSTREAM_API_KEY=gsk_...
BIG_MODEL=llama-3.3-70b-versatile
SMALL_MODEL=llama-3.1-8b-instant
```

### OpenRouter (access 100+ models through one key)
```env
UPSTREAM_BASE_URL=https://openrouter.ai/api
UPSTREAM_API_KEY=sk-or-...
BIG_MODEL=anthropic/claude-opus-4
SMALL_MODEL=google/gemini-flash-1.5
```

### Ollama (local, free)
```env
UPSTREAM_BASE_URL=http://localhost:11434
UPSTREAM_API_KEY=ollama
BIG_MODEL=qwen2.5-coder:32b
SMALL_MODEL=qwen2.5-coder:7b
```

---

## GitHub Actions deployment

The included workflow runs the proxy on a GitHub-hosted runner and exposes it publicly via [ngrok](https://ngrok.com) (free tier).

### Setup

1. Push this repo to GitHub (public repo = unlimited free minutes).

2. Add **Repository Secrets** (`Settings → Secrets → Actions`):

   | Secret | Value |
   |---|---|
   | `UPSTREAM_API_KEY` | Your provider API key |
   | `NGROK_AUTHTOKEN` | From [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) |
   | `UPSTREAM_BASE_URL` | *(optional)* Override provider URL |
   | `BIG_MODEL` | *(optional)* Override big model |
   | `SMALL_MODEL` | *(optional)* Override small model |
   | `REASONING_MODEL` | *(optional)* For extended thinking |

3. Trigger the workflow:
   ```
   GitHub → Actions → "Run Proxy" → Run workflow
   ```
   You can also override models per-run from the UI.

4. Watch the **"Print public URL"** step output:
   ```
   ╔══════════════════════════════════════════════════════════╗
   ║  PROXY URL → https://abc123.ngrok-free.app              ║
   ╚══════════════════════════════════════════════════════════╝
   ```

5. Use it:
   ```bash
   export ANTHROPIC_BASE_URL=https://abc123.ngrok-free.app
   export ANTHROPIC_API_KEY=dummy
   claude
   ```

### Limits

- GitHub Actions: max **6 hours** per job. The workflow runs for 5 hours by default.
- ngrok free tier: 1 concurrent tunnel, HTTPS only (which is fine).
- GitHub Student Pack gives you **unlimited minutes on public repos** and extra minutes on private repos.

### Tip: make it permanent

For a always-on proxy without the 6h limit, set up a **self-hosted runner** on any machine you control (VPS, Raspberry Pi, your home server):

```bash
# On your server
# Follow: https://github.com/YOUR_REPO/settings/actions/runners
# Then use  runs-on: self-hosted  in the workflow
```

---

## Architecture

```
Claude Code
    │  POST /v1/messages  (Anthropic format)
    ▼
┌────────────────────────────────────────┐
│           Proxy  (FastAPI)             │
│                                        │
│  1. Parse Anthropic payload            │
│  2. Resolve model name                 │
│  3. Convert messages / tools / images  │
│  4. Stream or batch request upstream   │
│  5. Convert response back to Anthropic │
└────────────────────────────────────────┘
    │  POST /v1/chat/completions  (OpenAI format)
    ▼
Any OpenAI-compatible endpoint
```

---

## Running tests

```bash
uv run pytest tests/ -v
```
