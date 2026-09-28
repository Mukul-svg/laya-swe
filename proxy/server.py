import os
import sys

# Ensure project root is in sys.path when executed directly as python proxy/server.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Optional, Dict, Any, Tuple
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from proxy.compactor import ContextCompactor, LayaSweCompactor, AxiomCompactor
    from proxy.memory_engine import LayaSweMemoryEngine, AxiomMemoryEngine, SessionMemoryManager
except ImportError:
    from compactor import ContextCompactor, LayaSweCompactor, AxiomCompactor
    from memory_engine import LayaSweMemoryEngine, AxiomMemoryEngine, SessionMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("laya_swe_proxy")

app = FastAPI(title="LAYA-SWE Memory & Compaction Proxy", version="1.0.0")

# Telemetry Log Path
TELEMETRY_LOG = os.path.join(PROJECT_ROOT, "benchmark_telemetry.jsonl")

def load_local_env():
    """Load local environment variables from .env if present without external dependencies."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception as e:
            logger.warning(f"Could not load local .env: {e}")

load_local_env()

# LAYA-SWE Multi-Session Memory Manager & Legacy Singleton Fallback
session_manager = SessionMemoryManager()
memory_engine = LayaSweMemoryEngine()
compactor = ContextCompactor(memory_engine=memory_engine, max_tail_messages=8)

def get_session_compactor(session_id: str, controller_mode: Optional[str] = None):
    """Retrieve or allocate an isolated memory engine and compactor for a session."""
    return session_manager.get_or_create(
        session_id=session_id,
        compactor_factory=lambda eng: ContextCompactor(memory_engine=eng, max_tail_messages=8),
        controller_mode=controller_mode
    )

# Async Locks for Concurrency Protection
telemetry_lock = asyncio.Lock()
stats_lock = asyncio.Lock()
config_lock = asyncio.Lock()

# Dynamic Benchmark State
current_benchmark_config = {
    "compaction": True,
    "task_id": "default_task"
}

# In-memory stats
stats = {
    "total_requests": 0,
    "total_raw_tokens": 0,
    "total_sent_tokens": 0,
    "total_tokens_saved": 0,
    "total_cached_tokens": 0,
    "runs": []
}


def get_upstream_target():
    """Resolve upstream URL, API key, and provider name dynamically with zero hardcoded credentials."""
    load_local_env()
    custom_url = os.getenv("UPSTREAM_URL", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()

    # 1. Custom explicit UPSTREAM_URL
    if custom_url:
        return custom_url.rstrip("/"), (groq_key or openai_key or openrouter_key), "custom"

    # 2. Genuine OpenAI key (ignore placeholder 'dummy')
    if openai_key and openai_key.lower() not in ("dummy", "placeholder", "none", "") and len(openai_key) > 15:
        return "https://api.openai.com/v1", openai_key, "openai"

    # 3. Groq Frontier LLM Provider (Ultra-fast, zero 402 credit blocks)
    if groq_key:
        return "https://api.groq.com/openai/v1", groq_key, "groq"

    # 4. OpenRouter fallback
    if openrouter_key:
        return "https://openrouter.ai/api/v1", openrouter_key, "openrouter"

    return "https://api.groq.com/openai/v1", "", "groq"


def _append_telemetry_file(entry_dict: dict):
    """Synchronous file append worker executed in separate thread."""
    os.makedirs(os.path.dirname(TELEMETRY_LOG), exist_ok=True)
    with open(TELEMETRY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry_dict) + "\n")


async def record_telemetry(
    task_id: str,
    compaction: bool,
    raw_tok: int,
    sent_tok: int,
    duration_s: float,
    upstream_model: str = "openai/gpt-oss-120b",
    cached_tokens: int = 0
):
    saved_tok = max(0, raw_tok - sent_tok)
    entry = {
        "timestamp": time.time(),
        "task_id": task_id,
        "mode": "with_system_1" if compaction else "control_full_context",
        "controller_mode": current_benchmark_config.get("controller_mode", memory_engine.controller.mode) if compaction else "control",
        "upstream_model": upstream_model,
        "raw_prompt_tokens": raw_tok,
        "sent_prompt_tokens": sent_tok,
        "tokens_saved": saved_tok,
        "savings_percentage": round((saved_tok / max(1, raw_tok)) * 100, 2),
        "cached_tokens": cached_tokens,
        "cache_hit_rate": round((cached_tokens / max(1, sent_tok)) * 100, 2),
        "latency_seconds": round(duration_s, 3)
    }
    async with stats_lock:
        stats["total_requests"] += 1
        stats["total_raw_tokens"] += raw_tok
        stats["total_sent_tokens"] += sent_tok
        stats["total_tokens_saved"] += saved_tok
        stats.setdefault("total_cached_tokens", 0)
        stats["total_cached_tokens"] += cached_tokens
        stats["runs"].append(entry)

    async with telemetry_lock:
        await asyncio.to_thread(_append_telemetry_file, entry)

    logger.info(f"[Telemetry] Task: {task_id} | Model: {upstream_model} | Mode: {entry['mode']} | Raw: {raw_tok} | Sent: {sent_tok} | Saved: {saved_tok} ({entry['savings_percentage']}%) | Cached: {cached_tokens}")


@app.get("/health")
async def health():
    upstream_url, actual_key, provider = get_upstream_target()
    masked_key = (actual_key[:6] + "..." + actual_key[-4:]) if len(actual_key) > 10 else "[UNSET]"
    return {
        "status": "ok",
        "engine": "LAYA-SWE",
        "system_one_controller": "LAYA-ModernBERT-421M",
        "system_one_mode": memory_engine.controller.mode,
        "laya_loaded": memory_engine.controller._laya_agent is not None,
        "upstream_provider": provider,
        "upstream_url": upstream_url,
        "upstream_key": masked_key,
        "memory_nodes": len(memory_engine.nodes_map),
        "graph_stats": memory_engine.get_stats(),
        "current_config": current_benchmark_config
    }


@app.get("/metrics")
async def get_metrics():
    async with stats_lock:
        return dict(stats)


@app.post("/reset")
async def reset_memory(request: Request = None):
    session_id = None
    if request:
        try:
            body = await request.json()
            session_id = body.get("session_id")
        except Exception:
            pass
    session_manager.reset_session(session_id)
    memory_engine.clear()
    compactor._ingested_message_ids.clear()
    return {"status": "memory_cleared", "session_id": session_id or "all"}


@app.post("/ingest")
async def ingest_observation(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    obs = data.get("observation", "")
    metadata = data.get("metadata", {})
    role = data.get("role") or metadata.get("role")
    node_id = memory_engine.ingest(obs, metadata=metadata, role=role)
    return {"status": "ok", "node_id": node_id}


@app.post("/query")
async def query_memory(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    query_text = data.get("query", "")
    top_k = int(data.get("top_k", 5))
    evidence = memory_engine.retrieve(query_text, top_k=top_k)
    return {"status": "ok", "evidence": evidence}


@app.post("/debug/classify")
async def debug_classify(request: Request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    text = data.get("text", "")
    meta = {}
    plane, is_pinned = memory_engine.controller.classify_plane(text, meta)
    return {
        "text": text[:200],
        "plane": plane.value if hasattr(plane, "value") else str(plane),
        "is_pinned": is_pinned,
        "laya": meta.get("laya", {}),
        "symbols": memory_engine.controller.extract_symbols(text)
    }


@app.post("/benchmark/configure")
async def configure_benchmark(request: Request):
    data = await request.json()
    async with config_lock:
        if "compaction" in data:
            current_benchmark_config["compaction"] = bool(data["compaction"])
        if "task_id" in data:
            current_benchmark_config["task_id"] = str(data["task_id"])
        if "controller_mode" in data:
            mode = str(data["controller_mode"])
            current_benchmark_config["controller_mode"] = mode
            memory_engine.controller.mode = mode
    logger.info(f"[Config] Benchmark configured: {current_benchmark_config}")
    return {"status": "ok", "config": current_benchmark_config}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    start_time = time.time()
    body = await request.json()

    # Determine compaction & task_id (priority: explicit header > benchmark config > env fallback)
    enable_compaction_hdr = request.headers.get("X-Benchmark-Compaction")
    if enable_compaction_hdr is not None:
        enable_compaction = str(enable_compaction_hdr).lower() in ("true", "1", "yes")
    else:
        async with config_lock:
            enable_compaction = current_benchmark_config.get("compaction", True)

    task_id_hdr = request.headers.get("X-Task-ID")
    if task_id_hdr:
        task_id = task_id_hdr
    else:
        async with config_lock:
            task_id = current_benchmark_config.get("task_id", "default_task")

    session_id = request.headers.get("X-Session-ID") or task_id
    active_controller_mode = current_benchmark_config.get("controller_mode")
    session_engine, session_compactor = get_session_compactor(session_id, controller_mode=active_controller_mode)

    raw_messages = body.get("messages", [])
    # Offload compaction computation to background worker thread to keep FastAPI event-loop non-blocking!
    compacted_messages, raw_tok, sent_tok = await asyncio.to_thread(
        session_compactor.process_and_compact,
        raw_messages,
        enable_compaction=enable_compaction
    )
    body["messages"] = compacted_messages

    # Resolve Upstream Target & Model Mapping
    upstream_url, actual_key, provider = get_upstream_target()

    # Route model according to provider
    if provider == "groq":
        requested_model = body.get("model", "")
        if requested_model and (requested_model.startswith("openai/gpt-oss-") or requested_model.startswith("qwen/")):
            body["model"] = requested_model
        else:
            body["model"] = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    elif provider == "openrouter":
        current_model = body.get("model", "")
        if current_model in ("gpt-4o-mini", "gpt-4o", "o1", "o3-mini", "gpt-4-turbo"):
            body["model"] = f"openai/{current_model}"

    forward_headers = {
        "Authorization": f"Bearer {actual_key}",
        "Content-Type": "application/json"
    }

    client = httpx.AsyncClient(timeout=120.0)

    # Handle streaming vs non-streaming
    is_stream = body.get("stream", False)

    # 1. Context Ceiling Detection: Prevent provider rate limit lockouts
    if sent_tok >= 7800:
        logger.warning(f"[Context Ceiling] Prompt size {sent_tok} exceeds provider 7800 TPM ceiling.")
        err_msg = {
            "error": {
                "message": f"Context ceiling exceeded: prompt has {sent_tok} tokens, exceeding provider 7800 TPM limit. Compaction is required to continue.",
                "type": "context_length_exceeded",
                "code": "context_length_exceeded"
            }
        }
        await record_telemetry(task_id, enable_compaction, raw_tok, sent_tok, time.time() - start_time)
        return JSONResponse(status_code=413, content=err_msg)

    # 2. Dynamic safe max_tokens to guarantee prompt + max_tokens <= 7850
    safe_max_tokens = max(768, min(body.get("max_tokens", 2048), 7850 - sent_tok))

    # Always request complete JSON from upstream to ensure atomic schema normalization
    upstream_body = dict(body)
    upstream_body["stream"] = False
    upstream_body["max_tokens"] = safe_max_tokens

    max_retries = 6
    for attempt in range(max_retries):
        try:
            resp = await client.post(
                f"{upstream_url}/chat/completions",
                json=upstream_body,
                headers=forward_headers
            )
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout) as conn_err:
            logger.warning(f"[Proxy Retry {attempt+1}/{max_retries}] Network connection issue: {conn_err}. Retrying in 2.0s...")
            await asyncio.sleep(2.0)
            continue

        is_tool_glitch = (resp.status_code == 400 and "tool_use_failed" in resp.text)
        if (resp.status_code in (402, 429, 502, 503) or is_tool_glitch) and attempt < max_retries - 1:
            if is_tool_glitch:
                logger.warning(f"[Proxy Retry {attempt+1}/{max_retries}] Model tool JSON syntax glitch. Retrying immediately...")
                await asyncio.sleep(0.5)
                continue
            err_str = resp.text
            retry_delay = 10.0
            try:
                if "retry-after" in resp.headers:
                    retry_delay = max(2.0, float(resp.headers["retry-after"]) + 0.5)
                else:
                    import re
                    m = re.search(r"try again in ([0-9\.]+)s", err_str)
                    if m:
                        retry_delay = float(m.group(1)) + 0.5
            except Exception:
                retry_delay = 10.0

            # Strictly maintain requested model across retries - zero silent model downgrades!
            # Continuous token bucket replenishment: sleep up to 30s per retry cycle
            sleep_duration = min(retry_delay, 30.0)
            logger.warning(f"[Proxy Retry {attempt+1}/{max_retries}] HTTP {resp.status_code} from {provider} on {upstream_body.get('model')}. Waiting {sleep_duration:.2f}s (requested: {retry_delay:.1f}s)...")
            await asyncio.sleep(sleep_duration)
            continue

        if resp.status_code >= 400:
            logger.error(f"[Upstream Error] HTTP {resp.status_code} from {provider}: {resp.text}")
            await client.aclose()
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

        await client.aclose()
        duration = time.time() - start_time

        # Structured Normalization on Complete JSON
        try:
            data = resp.json()
        except Exception as e:
            logger.error(f"[Upstream JSON Error] Could not decode upstream response as JSON: {e}")
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

        upstream_model_executed = data.get("model", upstream_body.get("model", "unknown"))
        usage_data = data.get("usage", {})
        prompt_details = usage_data.get("prompt_tokens_details", {})
        cached_tok = 0
        if isinstance(prompt_details, dict):
            cached_tok = prompt_details.get("cached_tokens", 0)
        if not cached_tok:
            cached_tok = usage_data.get("cached_tokens", 0)

        await record_telemetry(
            task_id,
            enable_compaction,
            raw_tok,
            sent_tok,
            duration,
            upstream_model=upstream_model_executed,
            cached_tokens=cached_tok
        )

        try:
            for choice in data.get("choices", []):
                msg = choice.get("message", {})
                if not msg.get("content") and msg.get("reasoning"):
                    msg["content"] = msg["reasoning"]
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    if fn_name in ("exec", "execute", "terminal", "sh", "run_command"):
                        fn["name"] = "bash"
                        fn_name = "bash"

                    args_str = fn.get("arguments", "")
                    if isinstance(args_str, str) and args_str.strip().startswith("{"):
                        try:
                            args_dict = json.loads(args_str)
                            if fn_name == "bash":
                                if "cmd" in args_dict and "command" not in args_dict:
                                    args_dict["command"] = args_dict.pop("cmd")
                            elif fn_name == "read":
                                if "line_start" in args_dict and "offset" not in args_dict:
                                    args_dict["offset"] = args_dict.pop("line_start")
                                if "line_end" in args_dict and "limit" not in args_dict:
                                    line_end = args_dict.pop("line_end")
                                    start_offset = args_dict.get("offset", 1)
                                    args_dict["limit"] = max(1, line_end - start_offset + 1)
                            elif fn_name == "edit":
                                if ("old_str" in args_dict or "oldText" in args_dict or "old_string" in args_dict) and "edits" not in args_dict:
                                    old_t = args_dict.pop("old_str", None) or args_dict.pop("oldText", None) or args_dict.pop("old_string", None)
                                    new_t = args_dict.pop("new_str", None) or args_dict.pop("newText", None) or args_dict.pop("new_string", None)
                                    args_dict["edits"] = [{"oldText": old_t, "newText": new_t or ""}]
                            elif fn_name == "write":
                                if "text" in args_dict and "content" not in args_dict:
                                    args_dict["content"] = args_dict.pop("text")
                                elif "body" in args_dict and "content" not in args_dict:
                                    args_dict["content"] = args_dict.pop("body")
                                elif "contents" in args_dict and "content" not in args_dict:
                                    args_dict["content"] = args_dict.pop("contents")

                            fn["arguments"] = json.dumps(args_dict)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Could not normalize tool calls in upstream JSON: {e}")

        if is_stream:
            async def stream_generator() -> AsyncGenerator[bytes, None]:
                chunk1 = {
                    "id": data.get("id", "chatcmpl-stream"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": data.get("model", "gpt-4o-mini"),
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk1)}\n\n".encode("utf-8")

                first_choice = data.get("choices", [{}])[0] if data.get("choices") else {}
                msg = first_choice.get("message", {})
                delta = {}
                if msg.get("content"):
                    delta["content"] = msg["content"]
                if msg.get("tool_calls"):
                    delta["tool_calls"] = msg["tool_calls"]

                finish = first_choice.get("finish_reason", "stop")
                chunk2 = {
                    "id": data.get("id", "chatcmpl-stream"),
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": data.get("model", "gpt-4o-mini"),
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
                }
                yield f"data: {json.dumps(chunk2)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            resp_headers = {"X-Upstream-Model": str(upstream_model_executed)}
            return StreamingResponse(stream_generator(), status_code=200, media_type="text/event-stream", headers=resp_headers)
        else:
            resp_headers = {"X-Upstream-Model": str(upstream_model_executed)}
            return Response(content=json.dumps(data).encode("utf-8"), status_code=200, media_type="application/json", headers=resp_headers)


if __name__ == "__main__":
    import uvicorn
    upstream_url, actual_key, provider = get_upstream_target()
    masked_key = (actual_key[:6] + "..." + actual_key[-4:]) if len(actual_key) > 10 else "[UNSET]"
    logger.info(f"[Auth] Upstream Provider: {provider.upper()} | Endpoint: {upstream_url} | Key: {masked_key}")
    port = int(os.getenv("PROXY_PORT", "8080"))
    uvicorn.run(app, host="127.0.0.1", port=port)
