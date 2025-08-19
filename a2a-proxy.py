"""
A2A SSE Proxy Launcher component for Langflow.
Full version (2025-08-09): JSON-RPC (POST /, /a2a) + SSE bridge to Langflow.
"""
from __future__ import annotations

import json
import re
import socket
import threading
import time
from typing import Any, AsyncIterator, Dict, Optional, Tuple
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from langflow.custom.custom_component.component import Component
from langflow.inputs import BoolInput, IntInput, MessageTextInput, SecretStrInput
from langflow.io import HandleInput, Output

import asyncio, logging
log = logging.getLogger("a2a-proxy")

# ---------------------------------------------------------------------------
# 🔧 Low-level utils
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """Try a bare TCP connect to check if a port is open."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _ping(base_url: str, timeout: float = 0.5) -> bool:
    """`/a2a/ping` health-check helper."""
    try:
        r = httpx.get(f"{base_url}/a2a/ping", timeout=timeout)
        return r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        return False


def _key_of(lf_url: str, api_key: str) -> str:
    return f"{lf_url.rstrip('/')}_{api_key[:6]}"


# ---------------- Langflow Flow helpers ----------------

def _flows_search(lf_url: str, api_key: str, query: Optional[str] = None) -> list:
    """Call Langflow `/api/v1/flows` with optional search."""
    url = f"{lf_url.rstrip('/')}/api/v1/flows"
    if query:
        url += f"?search={query}"
    r = httpx.get(url, headers={"x-api-key": api_key}, timeout=8)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _resolve_flow_id_sync(
    lf_url: str,
    api_key: str,
    flow_id: str,
    flow_name: str,
    allow_singleton_pick: bool = True,
) -> Tuple[str, str]:
    """Best-effort flow resolver (ID > name > singleton)."""
    if flow_id:
        try:
            items = _flows_search(lf_url, api_key)
            hit = next((x for x in items if str(x.get("id")) == str(flow_id)), None)
            if hit:
                return str(hit["id"]), hit.get("name") or flow_name or ""
        except Exception:
            pass
        return flow_id, flow_name or ""

    if flow_name:
        items = _flows_search(lf_url, api_key, query=flow_name)
        exact = next((x for x in items if x.get("name") == flow_name), None)
        target = exact or (items[0] if items else None)
        if target and target.get("id"):
            return str(target["id"]), target.get("name") or flow_name
        raise ValueError(f"'{flow_name}' 이름으로 flow_id를 찾지 못했습니다.")

    if allow_singleton_pick:
        items = _flows_search(lf_url, api_key)
        if len(items) == 1:
            return str(items[0]["id"]), items[0].get("name") or ""
        uniq_names = {x.get("name") for x in items if x.get("id")}
        if len(uniq_names) == 1 and items:
            return str(items[0]["id"]), items[0].get("name") or ""

    raise ValueError("flow_id/flow_name이 모두 비어있고 자동 선택도 실패했습니다.")


# ---------------- agent_card parsing helper ----------------

def _parse_agent_card(value: Any) -> Optional[Dict[str, Any]]:
    """Attempt to coerce incoming HandleInput payload to dict."""
    try:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            d = value.model_dump()
            txt = d.get("text")
            if isinstance(txt, str):
                try:
                    return json.loads(txt)
                except Exception:
                    return {"raw": txt}
            return d
        if hasattr(value, "text") and isinstance(getattr(value, "text"), str):
            txt = getattr(value, "text")
            try:
                return json.loads(txt)
            except Exception:
                return {"raw": txt}
        if hasattr(value, "data") and isinstance(getattr(value, "data"), dict):
            return getattr(value, "data")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return {"raw": value}
    except Exception:
        pass
    return None

# ---------------------------------------------------------------------------
# 🔧 SSE / JSON-RPC helpers
# ---------------------------------------------------------------------------

def _sse_event(data: Any, event: Optional[str] = None, id_: Optional[str] = None) -> bytes:
    """Serialize JSON/str to SSE bytes."""
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False)
    lines = []
    if event:
        lines.append(f"event: {event}")
    if id_:
        lines.append(f"id: {id_}")
    for line in (data.splitlines() or [""]):
        lines.append(f"data: {line}")
    lines.append("")  # end of event
    return ("\n".join(lines) + "\n").encode()


def _normalize_input_type(raw: Optional[str]) -> str:
    if not raw:
        return "any"
    raw = raw.lower()
    if raw in {"chat", "text", "any"}:
        return raw
    if raw.startswith("text/"):
        return "text"
    return "any"


async def _raise_for_upstream(resp: httpx.Response) -> None:
    """Convert Langflow 4xx/5xx to FastAPI HTTPException (bytes-safe)."""
    if resp.status_code < 400:
        return
    content = await resp.aread()
    try:
        detail = json.loads(content)
    except Exception:
        detail = content.decode("utf-8", errors="replace")
    raise HTTPException(status_code=resp.status_code, detail=detail)


# ---------------------------------------------------------------------------
# 🚀 FastAPI proxy app factory
# ---------------------------------------------------------------------------

def _sse_proxy_app(
    lf_url: str,
    api_key: str,
    flow_id_cfg: str,
    flow_name_cfg: str,
    stream_path: str,
    prefer_session_as_flow: bool,
    auto_pick_singleton: bool,
    resolved_flow_cache: Dict[str, Tuple[str, str]],
    agent_card_payload: Optional[Dict[str, Any]] = None,
) -> FastAPI:
    """Return a fully-wired FastAPI proxy app (JSON-RPC + SSE)."""

    app = FastAPI(
        title="A2A SSE Proxy (embedded)",
        docs_url=None,
        redoc_url=None,
    )

    # ------------ Request models ------------

    class A2AInput(BaseModel):
        type: str
        data: str

    class A2AReq(BaseModel):
        sender: str
        input: A2AInput
        mode: str
        session_id: Optional[str] = None
        output_type: Optional[str] = None
        input_type: Optional[str] = None
        variables: Optional[Dict[str, str]] = None

    # ------------ helpers ------------

    def _hdr() -> Dict[str, str]:
        return {
            "x-api-key": api_key,
            "content-type": "application/json",
            "accept": "text/event-stream, application/json",
            "connection": "keep-alive",
            "cache-control": "no-cache",
        }

    async def _resolve_for_request(req_session_id: Optional[str]) -> Tuple[str, str]:
        """Resolve flow for each request (cache aware)."""
        if flow_id_cfg:
            return flow_id_cfg, flow_name_cfg or ""
        if prefer_session_as_flow and req_session_id:
            return req_session_id, "(session-as-flow)"
        cache_key = _key_of(lf_url, api_key)
        if cache_key in resolved_flow_cache:
            return resolved_flow_cache[cache_key]
        rid, rname = _resolve_flow_id_sync(
            lf_url,
            api_key,
            flow_id_cfg,
            flow_name_cfg,
            allow_singleton_pick=auto_pick_singleton,
        )
        resolved_flow_cache[cache_key] = (rid, rname or "")
        return rid, rname or ""

    def _message_to_text(msg: dict) -> str:
        """
        A2A Message → text 추출(방어적).
        1) parts[*].text  또는  parts[*].root.text
        2) (실수 대비) arts[*].text  또는  arts[*].root.text
        3) 톱레벨 msg["text"] / msg["content"] / msg["value"]
        4) msg 자체가 str이면 그대로
        """
        if not msg:
            return ""
        if isinstance(msg, str):
            return msg.strip()

        texts: list[str] = []

        def collect_from_list(arr):
            for p in arr:
                if isinstance(p, dict):
                    # 표준
                    if p.get("kind") == "text" and isinstance(p.get("text"), str):
                        t = p["text"].strip()
                        if t:
                            texts.append(t)
                    else:
                        root = p.get("root")
                        if isinstance(root, dict) and isinstance(root.get("text"), str):
                            t = root["text"].strip()
                            if t:
                                texts.append(t)

        parts = msg.get("parts")
        if isinstance(parts, list):
            collect_from_list(parts)

        # 흔한 오타/변형
        arts = msg.get("arts")
        if isinstance(arts, list):
            collect_from_list(arts)

        for k in ("text", "content", "value"):
            v = msg.get(k)
            if isinstance(v, str) and v.strip():
                texts.append(v.strip())

        return "\n".join(texts).strip()


    def _rpc_envelope(rpc_id: str, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

    def _a2a_status_text(text: str) -> dict:
        return {
            "kind": "status-update",
            "message": {
                "message_id": str(uuid4()),
                "role": "agent",
                "parts": [{"kind": "text", "text": text}],
            },
        }

    def _a2a_message_text(text: str) -> dict:
        return {
            "kind": "message",
            "messageId": str(uuid4()),   # ← camelCase
            "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        }

    def _url_for_flow(flow_id: str, want_stream: bool) -> str:
        path = stream_path.format(flow_id=flow_id)
        if "stream=" in path:
            path = re.sub(r"stream=\w+", f"stream={'true' if want_stream else 'false'}", path)
        else:
            sep = "&" if "?" in path else "?"
            path = f"{path}{sep}stream={'true' if want_stream else 'false'}"
        return f"{lf_url.rstrip('/')}{path}"

    # 교체: 업스트림 SSE → (1) chunk별 delta 이벤트(JSON-RPC 랩핑) (2) 종료 시 full 텍스트 1회 추가 송출
    async def _jsonrpc_stream_bridge(upstream: httpx.Response, rpc_id: str) -> AsyncIterator[bytes]:
        final_buf: list[str] = []

        try:
            async for chunk in upstream.aiter_raw():
                if not chunk:
                    continue

                # 1) 디코드 및 CRLF 정규화
                try:
                    text = chunk.decode("utf-8")
                except Exception:
                    text = chunk.decode("utf-8", "replace")
                text = text.replace("\r\n", "\n")

                # 2) 라인 단위로 가볍게 SSE 보일러플레이트 제거
                #    - data: xxx    → xxx
                #    - event:/id:/코멘트(: ...) 라인은 무시
                #    - [DONE] 같은 종료 센티넬은 무시
                cleaned_lines: list[str] = []
                for line in text.split("\n"):
                    ls = line.lstrip()
                    if not ls:
                        continue
                    if ls.startswith(":"):  # SSE comment line
                        continue
                    if ls.lower().startswith("event:") or ls.lower().startswith("id:"):
                        continue
                    if ls[:5].lower() == "data:":
                        payload = ls[5:].lstrip()
                        if payload.strip() in {"[DONE]", ""}:
                            continue
                        cleaned_lines.append(payload)
                    else:
                        # NDJSON 등 비표준도 최대한 통과
                        cleaned_lines.append(line)

                piece = "\n".join(cleaned_lines)
                if not piece:
                    continue

                # 3) chunk별 delta 이벤트 전송(A2A 메시지 규격 + JSON-RPC 랩핑)
                final_buf.append(piece)
                out_delta = _rpc_envelope(rpc_id, _a2a_message_text(piece))
                yield _sse_event(out_delta)

        except (asyncio.CancelledError, BrokenPipeError, ConnectionResetError):
            # 다운스트림 종료 → 조용히 정리
            return
        except Exception as e:
            # 에러를 메시지로 알리고 종료(원하면 로그만 남기고 return 해도 됨)
            err = _rpc_envelope(rpc_id, _a2a_message_text(f"[proxy error] {e}"))
            yield _sse_event(err)
            return

        # 4) 스트림 종료 시 "완성 텍스트" 1회 추가 송출
        if final_buf:
            full = "".join(final_buf)
            # 연속 개행 정리(선택)
            full = re.sub(r"\n{3,}", "\n\n", full).strip()
            if full:
                out_final = _rpc_envelope(rpc_id, _a2a_message_text(full))
                yield _sse_event(out_final)


    def _guess_output_text(data: Any) -> str:
        """Langflow 비스트리밍 응답에서 텍스트 추출 (여러 케이스 방어)."""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for k in ("output_text", "text", "message", "result", "data"):
                v = data.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            outs = data.get("outputs")
            if isinstance(outs, list) and outs:
                cand = outs[0]
                if isinstance(cand, dict):
                    for k in ("text", "output_text", "result"):
                        v = cand.get(k)
                        if isinstance(v, str) and v.strip():
                            return v
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)
        
    # _sse_proxy_app(...) 내부, "helpers" 블록에 추가
    async def _raise_upstream_or_http(resp: httpx.Response) -> None:
        """Langflow 4xx/5xx → FastAPI HTTPException"""
        if resp.status_code < 400:
            return
        content = await resp.aread()
        try:
            detail = json.loads(content)
        except Exception:
            detail = content.decode("utf-8", errors="replace")
        raise HTTPException(status_code=resp.status_code, detail=detail)


    # ------------ Routes ------------

    @app.get("/a2a/ping")
    async def ping():
        try:
            rid, rname = await _resolve_for_request(None)
            return {"ok": True, "flow_id": rid, "flow_name": rname}
        except Exception as e:
            return {"ok": False, "flow_resolve_error": str(e)}

    @app.get("/a2a/agent-card")
    async def agent_card():
        rid, rname = "", ""
        try:
            rid, rname = await _resolve_for_request(None)
        except Exception:
            pass
        if not agent_card_payload:
            raise HTTPException(status_code=404, detail="Agent card not found")
        result = dict(agent_card_payload)
        result["flow"] = {"id": rid, "name": rname}
        return result

    # well-known: 루트와 /a2a 둘 다 제공 (클라 base_url 차이 대응)
    @app.get("/.well-known/agent-card.json", include_in_schema=False)
    async def agent_card_well_known_root():
        return await agent_card()

    @app.get("/a2a/.well-known/agent-card.json", include_in_schema=False)
    async def agent_card_well_known_prefixed():
        return await agent_card()


    # -------- JSON-RPC over HTTP (SDK 호환): POST /, /a2a --------

    def parse_and_get_data_text(s, *, require_nonempty: bool = False) -> Optional[str]:
        """
        payload를 JSON으로 파싱하고, 파싱된 객체에서 obj['data']['text']를 반환.
        - payload: str 또는 bytes (기타 타입은 str로 변환 시도)
        - require_nonempty=True면 공백뿐인 문자열은 None으로 취급
        """
        # # 1) 입력 정규화
        # if isinstance(payload, (bytes, bytearray)):
        #     s = payload.decode("utf-8", "ignore")
        # else:
        #     s = str(payload)

        # 2) JSON 파싱
        try:
            obj = json.loads(s)
        except Exception:
            return None

        # 3) data.text 추출
        if isinstance(obj, dict):
            data = obj.get("data")
            if isinstance(data, dict):
                text = data.get("text")
                if isinstance(text, str):
                    if require_nonempty and not text.strip():
                        return None
                    return text

        return None

    async def _handle_jsonrpc(request: Request):
        print("[proxy] hit", request.url.path, "accept=", request.headers.get("accept"))

        body = await request.json()
        rpc_id = body.get("id") or str(uuid4())
        params = body.get("params", {}) or {}

        want_stream = bool(params.get("stream")) or (
            "text/event-stream" in (request.headers.get("accept") or "")
        )

        # ① 표준 경로: params.message → text
        msg = params.get("message") or {}
        text_in = _message_to_text(msg)

        # ② 최후 폴백들: params.input.data / params.text / params.input_value
        if not text_in:
            inp = params.get("input") or {}
            if isinstance(inp, dict) and isinstance(inp.get("data"), str):
                text_in = inp["data"].strip()
        if not text_in and isinstance(params.get("text"), str):
            text_in = params["text"].strip()
        if not text_in and isinstance(params.get("input_value"), str):
            text_in = params["input_value"].strip()

        # (임시 디버그) 실제로 무엇을 넘기는지 확인
        print(f"[proxy] want_stream={want_stream} extracted='{text_in[:80]}'")

        # ← 여기까지 왔는데도 text_in이 비어있다면, 400으로 명확히 알려주도록 권장
        if not text_in:
            raise HTTPException(status_code=400, detail="No input text found in params.message/params.input/params.text")

        try:
            rid, rname = await _resolve_for_request(params.get("session_id") or params.get("context_id"))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"flow_id resolve 실패: {e}")

        # Langflow 호출 페이로드: 입력 타입은 기본을 'chat'로 잡는 편이 안전
        payload = {
            "input_value": text_in,
            "input_type": params.get("input_type") or "chat",   # ← 기본 chat
            "output_type": params.get("output_type") or "chat", # ← 기본 chat
            "tweaks": params.get("variables") or {},
            "session_id": params.get("session_id"),
        }

        url = _url_for_flow(rid, want_stream)
        print(f"[proxy→LF] POST {url} input_type={payload['input_type']} input_value[:40]={payload['input_value'][:40]!r}")

        if want_stream:
            async def gen():
                # --- helpers ---------------------------------------------------------
                def _extract_text_from_json_like(s: str) -> str:
                    """
                    s가 JSON이면 data.text(최우선) → result.data.text → text 순으로 추출.
                    JSON이 아니거나 경로가 없으면 원문 s를 반환.
                    """
                    try:
                        obj = json.loads(s)
                    except Exception:
                        return s

                    def _get_path(d, *keys):
                        cur = d
                        for k in keys:
                            if isinstance(cur, dict) and k in cur:
                                cur = cur[k]
                            else:
                                return None
                        return cur

                    for path in (("data", "text"), ("result", "data", "text"), ("text",)):
                        v = _get_path(obj, *path)
                        if isinstance(v, str) and v:
                            return v

                    return s

                # --------------------------------------------------------------------
                try:
                    headers = _hdr()
                    headers = dict(headers) if headers else {}
                    headers.setdefault("Accept", "text/event-stream")

                    async with httpx.AsyncClient(timeout=None, http2=False, headers=headers) as client:
                        async with client.stream("POST", url, json=payload) as upstream:
                            ctype = (upstream.headers.get("content-type") or "").lower()
                            te = (upstream.headers.get("transfer-encoding") or "").lower()
                            print("[proxy] upstream ctype=", ctype, "te=", te)

                            if upstream.status_code >= 400:
                                detail = await upstream.aread()
                                err = detail.decode("utf-8", "ignore")
                                out = _rpc_envelope(rpc_id, _a2a_message_text(f"[upstream error] {err}"))
                                yield _sse_event(out)
                                return

                            if "text/event-stream" in ctype:
                                # SSE: data: {...} 라인만 집어 JSON 파싱 후 data.text 추출
                                prev_text = ""  # 이전까지 받은 누적 문자열
                                async for line in upstream.aiter_lines():
                                    if not line:
                                        continue
                                    text = parse_and_get_data_text(line)
                                    if not text and text != "":
                                        continue

                                    # 길이가 줄었거나(재시작) 이전 prefix가 아니면 -> 전체를 델타로 간주
                                    if prev_text and text.startswith(prev_text):
                                        delta = text[len(prev_text):]
                                    else:
                                        delta = text
                                    
                                    prev_text = text

                                    if delta:
                                        out = _rpc_envelope(rpc_id, _a2a_message_text(delta))
                                        yield _sse_event(out)
                            else:
                                # NDJSON / line-delimited JSON: 각 라인을 JSON으로 보고 data.text 추출
                                async for line in upstream.aiter_lines():
                                    if not line or not line.strip():
                                        continue
                                    text = _extract_text_from_json_like(line)
                                    out = _rpc_envelope(rpc_id, _a2a_message_text(text))
                                    yield _sse_event(out)

                            # (선택) 정상 종료 꼬리표
                            # yield _sse_event(_rpc_envelope(rpc_id, _a2a_status_text("done")), event="end")

                except asyncio.CancelledError:
                    print("stream cancelled by client")
                except (BrokenPipeError, ConnectionResetError):
                    print("downstream closed")
                except Exception as e:
                    print(f"stream error: {e!r}")
                    out = _rpc_envelope(rpc_id, _a2a_message_text(f"[proxy error] {e}"))
                    yield _sse_event(out)

            return StreamingResponse(
                gen(),
                media_type="text/event-stream; charset=utf-8",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )


        # 🔽 여기부터 비-스트리밍(한방 JSON)
        async with httpx.AsyncClient(timeout=60, headers=_hdr()) as client:
            # want_stream=False로 다시 URL 보장
            url_nonstream = _url_for_flow(rid, False)
            r = await client.post(url_nonstream, json=payload)
            await _raise_upstream_or_http(r)
            data = r.json()
            out_text = _guess_output_text(data)
            return JSONResponse(_rpc_envelope(rpc_id, _a2a_message_text(out_text)))
        

    @app.post("/", include_in_schema=False)
    async def a2a_jsonrpc_root(request: Request):
        return await _handle_jsonrpc(request)

    @app.post("/a2a", include_in_schema=False)
    async def a2a_jsonrpc_alias(request: Request):
        return await _handle_jsonrpc(request)

    return app


# ---------------------------------------------------------------------------
# 🧩 Langflow Component
# ---------------------------------------------------------------------------

class ProxyLauncher(Component):
    display_name = "A2A SSE Proxy Launcher"
    description = (
        "Start (if not running) a local proxy that bridges A2A JSON-RPC/SSE to a Langflow flow. "
        "flow_id 자동 해석(flow_name/싱글톤/세션) 지원."
    )
    icon = "server"
    name = "A2AProxyLauncher"

    _server_threads: Dict[tuple, threading.Thread] = {}
    _resolved_flow: Dict[str, Tuple[str, str]] = {}
    _lock = threading.Lock()

    inputs = [
        BoolInput(name="enabled", display_name="Enable Launcher", value=True),
        MessageTextInput(name="host", display_name="Bind Host", value="0.0.0.0", show=False),
        IntInput(name="port", display_name="Bind Port", value=5008, show=False),

        MessageTextInput(name="langflow_url", display_name="Langflow URL", value="http://127.0.0.1:7860"),
        SecretStrInput(name="langflow_api_key", display_name="Langflow API Key", value=""),

        MessageTextInput(name="flow_name", display_name="Flow Name", value="", advanced=False),
        MessageTextInput(name="flow_id", display_name="Flow ID (optional)", value="", advanced=True),

        MessageTextInput(
            name="stream_path",
            display_name="Stream Path (format)",
            value="/api/v1/run/{flow_id}?stream=true",
            advanced=True
        ),
        BoolInput(
            name="prefer_session_as_flow",
            display_name="Use session_id as flow_id if available",
            value=False,
            advanced=True
        ),
        BoolInput(
            name="auto_pick_singleton",
            display_name="Auto-pick if workspace has a single flow",
            value=True,
            advanced=True
        ),
        HandleInput(  # Agent Card 입력(선택)
            name="agent_card",
            display_name="Agent Card",
            input_types=["Message"],
            is_list=False,
            advanced=False,
        ),
    ]

    outputs = [Output(display_name="Status", name="status", method="launch")]

    def launch(self):
        cls = self.__class__

        enabled = bool(getattr(self, "enabled", True))

        # 기본 bind host/port
        host = str(getattr(self, "host", "0.0.0.0") or "0.0.0.0")
        try:
            port = int(getattr(self, "port", 5008) or 5008)
        except Exception:
            port = 5008

        # Agent Card 입력 파싱 (선택)
        agent_card_raw = getattr(self, "agent_card", None)
        agent_card_payload = _parse_agent_card(agent_card_raw)

        # Agent Card의 url에서 포트만 우선 채택 (bind host는 0.0.0.0 유지 권장)
        if agent_card_payload and isinstance(agent_card_payload.get("url"), str):
            try:
                u = httpx.URL(agent_card_payload["url"])
                if u.port:
                    port = int(u.port)
            except Exception:
                pass

        langflow_url = str(getattr(self, "langflow_url", "")).rstrip("/")
        langflow_api_key = str(getattr(self, "langflow_api_key", ""))
        flow_id = str(getattr(self, "flow_id", "") or "")
        flow_name = str(getattr(self, "flow_name", "") or "")
        stream_path = str(getattr(self, "stream_path", "/api/v1/run/{flow_id}?stream=false"))
        prefer_session_as_flow = bool(getattr(self, "prefer_session_as_flow", False))
        auto_pick_singleton = bool(getattr(self, "auto_pick_singleton", True))

        base_url = f"http://{host}:{port}"

        if not enabled:
            return {"running": False, "base_url": base_url, "message": "Launcher disabled."}
        if not langflow_url or not langflow_api_key:
            return {"running": False, "base_url": base_url, "message": "Langflow URL/API Key required."}

        # 이미 떠 있으면 패스
        if _port_open(host, port) and _ping(base_url):
            return {"running": True, "base_url": base_url, "message": "Proxy already running."}

        # 같은 프로세스 내 재기동 방지
        key = (host, port)
        with cls._lock:
            if key in cls._server_threads and cls._server_threads[key].is_alive():
                return {"running": True, "base_url": base_url, "message": "Proxy thread already running."}

        # 미리 1회 flow 해석 시도
        try:
            rid, rname = _resolve_flow_id_sync(
                langflow_url, langflow_api_key, flow_id, flow_name, allow_singleton_pick=auto_pick_singleton
            )
            with cls._lock:
                cls._resolved_flow[_key_of(langflow_url, langflow_api_key)] = (rid, rname or "")
            pre_msg = f"flow resolved: id={rid}, name={rname}"
        except Exception as e:
            pre_msg = f"pre-resolve skipped: {e}"

        # FastAPI 앱 구성
        app = _sse_proxy_app(
            lf_url=langflow_url,
            api_key=langflow_api_key,
            flow_id_cfg=flow_id,
            flow_name_cfg=flow_name,
            stream_path=stream_path,
            prefer_session_as_flow=prefer_session_as_flow,
            auto_pick_singleton=auto_pick_singleton,
            resolved_flow_cache=cls._resolved_flow,
            agent_card_payload=agent_card_payload,
        )

        def _serve():
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            uvicorn.Server(config).run()

        th = threading.Thread(target=_serve, daemon=True, name=f"a2a-proxy-{host}:{port}")
        th.start()
        with cls._lock:
            cls._server_threads[key] = th

        # Healthcheck 대기
        for _ in range(50):
            if _port_open(host, port) and _ping(base_url):
                return {"running": True, "base_url": base_url, "message": f"Proxy started. {pre_msg}"}
            time.sleep(0.1)

        return {"running": False, "base_url": base_url, "message": "Proxy thread started but healthcheck failed."}
