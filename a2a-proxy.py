"""
A2A SSE Proxy Launcher component for Langflow.
Full version (2025-08-09): JSON-RPC (POST /) + SSE bridge to Langflow.
"""
from __future__ import annotations

import json
import re
import socket
import threading
import time
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from langflow.custom.custom_component.component import Component
from langflow.inputs import BoolInput, IntInput, MessageTextInput, SecretStrInput
from langflow.io import HandleInput, Output

import logging
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
    """`/ping` health-check helper."""
    try:
        r = httpx.get(f"{base_url}/ping", timeout=timeout)
        return r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        return False


def _key_of(lf_url: str, api_key: str) -> str:
    return f"{lf_url.rstrip('/')}_{api_key[:6]}"


# ---------------- A2A Discovery Service Integration ----------------

def _derive_agent_id(agent_card: Dict[str, Any]) -> str:
    """Agent ID 정규화: name/url 기반으로 소문자+숫자+대시/언더스코어만 허용."""
    raw = (agent_card.get("id") or agent_card.get("name") or agent_card.get("url") or "agent").strip()
    # 공백 → 대시, 비허용 문자 제거
    norm = re.sub(r"\s+", "-", raw)
    norm = re.sub(r"[^a-zA-Z0-9_-]", "", norm)
    if not norm:
        norm = "agent"
    return norm.lower()


def _register_agent_to_discovery(
    discovery_url: str,
    agent_card: Dict[str, Any],
    base_url: str,
    timeout: float = 10.0
) -> Tuple[bool, str]:
    """
    A2A Discovery 서비스에 에이전트를 등록합니다.

    - 사전 진단: GET / 로 프로토콜 확인, GET /agents 로 선점/충돌 여부 점검
    - 등록 실패(4xx/5xx/409 등) 시 상세 오류 메시지 로깅
    - 이미 존재 시 상태를 active 로 전환
    """
    try:
        if not discovery_url or not discovery_url.strip():
            return False, "Discovery URL이 설정되지 않음"
        discovery_url = discovery_url.rstrip('/')

        agent_id = _derive_agent_id(agent_card)
        agent_name = agent_card.get("name") or agent_id
        agent_endpoint = base_url

        # 사전 진단: discovery info
        try:
            info = httpx.get(f"{discovery_url}/", timeout=timeout)
            log.info("[discovery] info %s -> %s", info.status_code, info.text[:256])
        except Exception as e:
            log.warning("[discovery] info check failed: %s", e)

        # 사전 진단: existing agents
        existing = None
        try:
            r_agents = httpx.get(f"{discovery_url}/agents", timeout=timeout)
            if r_agents.status_code == 200:
                data = r_agents.json() if r_agents.headers.get("content-type", "").startswith("application/json") else {}
                if isinstance(data, dict) and isinstance(data.get("agents"), list):
                    for a in data["agents"]:
                        if str(a.get("id")) == agent_id:
                            existing = a
                            break
        except Exception as e:
            log.warning("[discovery] list_agents failed: %s", e)

        headers = {"Content-Type": "application/json"}
        # 등록 페이로드
        registration_data = {
            "id": agent_id,
            "name": agent_name,
            "endpoint": agent_endpoint,
        }
        log.info("[discovery] register payload: %s", json.dumps(registration_data, ensure_ascii=False))

        # 이미 존재하면 기존 에이전트로 처리
        if existing:
            return True, f"기존 에이전트가 이미 등록되어 있습니다: {agent_name} (Discovery 서비스가 자동으로 상태를 확인합니다)"

        # 신규 등록
        response = httpx.post(
            f"{discovery_url}/agent",
            json=registration_data,
            timeout=timeout,
            headers=headers,
        )
        log.info("[discovery] register resp %s: %s", response.status_code, response.text[:512])

        # 201/200 허용 - Discovery 서비스가 자동으로 ping하여 상태를 active로 변경
        if response.status_code in (200, 201):
            return True, f"에이전트가 성공적으로 등록되었습니다: {agent_name} (Discovery 서비스가 자동으로 상태를 확인합니다)"

        # 409(충돌) → 기존 에이전트로 처리
        if response.status_code == 409:
            return True, f"충돌 감지 → 기존 에이전트가 이미 등록되어 있습니다: {agent_name} (Discovery 서비스가 자동으로 상태를 확인합니다)"

        # 그 외 실패 → 상세 메시지
        return False, f"에이전트 등록 실패: HTTP {response.status_code} - {response.text}"

    except httpx.ConnectError:
        return False, f"Discovery 서비스에 연결할 수 없습니다: {discovery_url}"
    except httpx.TimeoutException:
        return False, f"Discovery 서비스 응답 시간 초과: {timeout}초"
    except Exception as e:
        return False, f"에이전트 등록 중 오류 발생: {str(e)}"


def _deactivate_agent_in_discovery(
    discovery_url: str,
    agent_card: Dict[str, Any],
    timeout: float = 10.0
) -> Tuple[bool, str]:
    """
    A2A Discovery 서비스에서 에이전트를 제거합니다.
    (Discovery 서비스에 상태 변경 API가 없으므로 제거 방식 사용)

    Args:
        discovery_url: Discovery 서비스의 기본 URL
        agent_card: Agent Card 정보
        timeout: 요청 타임아웃

    Returns:
        (성공여부, 메시지)
    """
    try:
        if not discovery_url or not discovery_url.strip():
            return False, "Discovery URL이 설정되지 않음"

        discovery_url = discovery_url.rstrip('/')
        agent_id = _derive_agent_id(agent_card)

        # Discovery 서비스에 DELETE API가 없으므로, 로그만 남기고 성공으로 처리
        log.info(f"[discovery] 에이전트 비활성화 요청: {agent_id} (Discovery 서비스가 자동으로 ping하여 상태를 확인합니다)")
        return True, f"에이전트 비활성화 요청이 기록되었습니다: {agent_id} (Discovery 서비스가 자동으로 상태를 확인합니다)"

    except Exception as e:
        return False, f"에이전트 비활성화 중 오류 발생: {str(e)}"


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

    # ------------ Flow resolution (once at startup) ------------
    # flow_id_cfg가 있으면 그것을 사용, 없으면 해결
    if flow_id_cfg:
        resolved_flow_id = flow_id_cfg
        resolved_flow_name = flow_name_cfg or ""
    else:
        # 캐시에서 먼저 확인
        cache_key = _key_of(lf_url, api_key)
        if cache_key in resolved_flow_cache:
            resolved_flow_id, resolved_flow_name = resolved_flow_cache[cache_key]
        else:
            # 해결하고 캐시에 저장
            resolved_flow_id, resolved_flow_name = _resolve_flow_id_sync(
                lf_url,
                api_key,
                flow_id_cfg,
                flow_name_cfg,
                allow_singleton_pick=auto_pick_singleton,
            )
            resolved_flow_cache[cache_key] = (resolved_flow_id, resolved_flow_name or "")

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


    def _message_to_text(msg: dict) -> str:
        """
        A2A Message → text 추출 (단순화).
        parts[0].text를 우선 사용, 실패 시 기본 폴백
        """
        if not msg:
            return ""
        if isinstance(msg, str):
            return msg.strip()

        # parts[0].text 우선 시도
        parts = msg.get("parts")
        if isinstance(parts, list) and parts:
            first_part = parts[0]
            if isinstance(first_part, dict):
                text = first_part.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        # 기본 폴백
        for k in ("text", "content", "value"):
            v = msg.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        return ""


    def _rpc_envelope(rpc_id: str, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


    def _a2a_message_text(text: str) -> dict:
        return {
            "kind": "message",
            "messageId": str(uuid4()),   # ← camelCase
            "role": "agent",
            "parts": [{"kind": "text", "text": text}],
        }

    def _extract_text_from_langflow_json(obj: Any) -> Optional[str]:
        """
        Langflow 응답 JSON에서 텍스트를 우선 경로로 추출.
        우선순위:
        1) outputs[0].outputs[0].results.message.data.text
        2) outputs[0].outputs[0].results.message.text
        찾지 못하면 None
        """
        try:
            outputs = obj.get("outputs")
            if isinstance(outputs, list) and outputs:
                outer0 = outputs[0]
                if isinstance(outer0, dict):
                    inner_outputs = outer0.get("outputs")
                    if isinstance(inner_outputs, list) and inner_outputs:
                        inner0 = inner_outputs[0]
                        if isinstance(inner0, dict):
                            results = inner0.get("results")
                            if isinstance(results, dict):
                                message = results.get("message")
                                if isinstance(message, dict):
                                    data = message.get("data")
                                    if isinstance(data, dict):
                                        t = data.get("text")
                                        if isinstance(t, str) and t:
                                            return t
                                    t2 = message.get("text")
                                    if isinstance(t2, str) and t2:
                                        return t2
        except Exception:
            pass
        return None

    def _url_for_flow(flow_id: str, want_stream: bool) -> str:
        path = stream_path.format(flow_id=flow_id)
        if "stream=" in path:
            path = re.sub(r"stream=\w+", f"stream={'true' if want_stream else 'false'}", path)
        else:
            sep = "&" if "?" in path else "?"
            path = f"{path}{sep}stream={'true' if want_stream else 'false'}"
        return f"{lf_url.rstrip('/')}{path}"

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

    @app.get("/ping")
    async def ping():
        """Discovery 서비스용 간단한 헬스체크 - HTTP 200만 반환"""
        # Discovery 서비스는 단순히 HTTP 200 응답만 확인하므로 복잡한 JSON 불필요
        return {"ok": True}

    @app.get("/health")
    async def health():
        """상세한 헬스체크 - flow 해석 포함"""
        return {
            "ok": True, 
            "status": "proxy_running",
            "flow_id": resolved_flow_id, 
            "flow_name": resolved_flow_name,
            "timestamp": time.time()
        }

    @app.post("/discovery/register")
    async def discovery_register():
        if not agent_card_payload:
            raise HTTPException(status_code=400, detail="Agent card not loaded")
        # base_url 는 실행 시점에서 역산하기 어렵기 때문에 클라이언트가 Host 헤더로 접근한 URL을 사용
        try:
            # request.url 은 path 포함이므로 netloc만 재구성
            # FastAPI에서 Request 주입
            from fastapi import Request as _Req
            async def _inner(req: _Req):
                scheme = (req.headers.get("x-forwarded-proto") or req.url.scheme)
                host = req.headers.get("x-forwarded-host") or req.headers.get("host") or "localhost"
                base_url = f"{scheme}://{host}"
                # Discovery URL은 런처 컴포넌트 설정에서만 알 수 있지만, 라우트에서는 접근 불가 → 400 안내
                return JSONResponse({"ok": False, "reason": "Discovery URL is configured in launcher only", "base_url_guess": base_url})
            return await _inner  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agent-card")
    async def agent_card():
        if not agent_card_payload:
            raise HTTPException(status_code=404, detail="Agent card not found")
        result = dict(agent_card_payload)
        result["flow"] = {"id": resolved_flow_id, "name": resolved_flow_name}
        return result

    # well-known: 표준 경로만 제공
    @app.get("/.well-known/agent-card.json", include_in_schema=False)
    async def agent_card_well_known_root():
        return await agent_card()




    # -------- JSON-RPC over HTTP (SDK 호환): POST / --------

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

        body = await request.json()
        rpc_id = body.get("id") or str(uuid4())
        params = body.get("params", {}) or {}

        # stream 파라미터 처리: "false", "0", false, 0 등은 False로 처리
        stream_param = params.get("stream")
        if isinstance(stream_param, str):
            stream_param = stream_param.lower() not in ("false", "0", "no", "off")
        elif isinstance(stream_param, (int, float)):
            stream_param = bool(stream_param)
        else:
            stream_param = bool(stream_param)
            
        want_stream = stream_param or (
            "text/event-stream" in (request.headers.get("accept") or "")
        )
        print(f"[proxy] want_stream={want_stream}, stream_param={stream_param}")

        # ① 표준 경로: params.message → text
        msg = params.get("message") or {}
        print(f"[proxy] msg={msg}")    
        
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
        print(f"[proxy] stream_param={params.get('stream')} want_stream={want_stream} extracted='{text_in[:80]}'")

        # ← 여기까지 왔는데도 text_in이 비어있다면, 400으로 명확히 알려주도록 권장
        if not text_in:
            raise HTTPException(status_code=400, detail="No input text found in params.message/params.input/params.text")

        # Flow ID 결정: session-as-flow 모드이거나 기본 해결된 값 사용
        req_session_id = params.get("session_id") or params.get("context_id")
        if prefer_session_as_flow and req_session_id:
            rid, rname = req_session_id, "(session-as-flow)"
        else:
            rid, rname = resolved_flow_id, resolved_flow_name

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
                    s가 JSON이면 Langflow 표준 경로 우선 시도, 실패 시 기본 폴백.
                    JSON이 아니면 원문 s를 반환.
                    """
                    try:
                        obj = json.loads(s)
                    except Exception:
                        return s

                    # Langflow 표준 outputs 경로 우선 시도
                    if isinstance(obj, dict):
                        t0 = _extract_text_from_langflow_json(obj)
                        if isinstance(t0, str) and t0:
                            return t0

                    # 기본 폴백: data.text, result.data.text, text 순
                    if isinstance(obj, dict):
                        for path in (("data", "text"), ("result", "data", "text"), ("text",)):
                            cur = obj
                            for k in path:
                                if isinstance(cur, dict) and k in cur:
                                    cur = cur[k]
                                else:
                                    cur = None
                                    break
                            if isinstance(cur, str) and cur:
                                return cur

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


                except (BrokenPipeError, ConnectionResetError):
                    print("downstream closed")
                except Exception as e:
                    if "CancelledError" in str(type(e)):
                        print("stream cancelled by client")
                    else:
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
            # Langflow 비스트리밍 응답에서 텍스트 추출 (우선 경로 → 폴백)
            out_text = None
            if isinstance(data, dict):
                out_text = _extract_text_from_langflow_json(data)
            if not out_text:
                if isinstance(data, str):
                    out_text = data
                elif isinstance(data, dict):
                    for k in ("output_text", "text", "message", "result", "data"):
                        v = data.get(k)
                        if isinstance(v, str) and v.strip():
                            out_text = v
                            break
                    else:
                        outs = data.get("outputs")
                        if isinstance(outs, list) and outs:
                            cand = outs[0]
                            if isinstance(cand, dict):
                                for k in ("text", "output_text", "result"):
                                    v = cand.get(k)
                                    if isinstance(v, str) and v.strip():
                                        out_text = v
                                        break
                                else:
                                    out_text = json.dumps(data, ensure_ascii=False)
                            else:
                                out_text = json.dumps(data, ensure_ascii=False)
                        else:
                            out_text = json.dumps(data, ensure_ascii=False)
                else:
                    try:
                        out_text = json.dumps(data, ensure_ascii=False)
                    except Exception:
                        out_text = str(data)
            return JSONResponse(_rpc_envelope(rpc_id, _a2a_message_text(out_text)))
        

    @app.post("/", include_in_schema=False)
    async def a2a_jsonrpc_root(request: Request):
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
    
    def __del__(self):
        """컴포넌트가 소멸될 때 discovery 서비스에서 에이전트 상태를 비활성화"""
        try:
            if hasattr(self, 'discovery_url') and hasattr(self, 'agent_card'):
                discovery_url = getattr(self, 'discovery_url', '')
                agent_card_raw = getattr(self, 'agent_card', None)
                if discovery_url and agent_card_raw:
                    agent_card_payload = _parse_agent_card(agent_card_raw)
                    if agent_card_payload:
                        _deactivate_agent_in_discovery(discovery_url, agent_card_payload)
        except Exception:
            pass  # 소멸자에서는 오류를 무시

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
        MessageTextInput(
            name="discovery_url",
            display_name="A2A Discovery Service URL",
            value="http://localhost:8000",
            advanced=True,
            info="A2A Discovery 서비스의 기본 URL (예: http://localhost:8000)"
        ),
        BoolInput(
            name="auto_register_discovery",
            display_name="Auto-register to Discovery Service",
            value=True,
            advanced=True,
            info="프록시 시작 시 자동으로 Discovery 서비스에 에이전트 등록"
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
        
        # Discovery 서비스 설정
        discovery_url = str(getattr(self, "discovery_url", "") or "").strip()
        auto_register_discovery = bool(getattr(self, "auto_register_discovery", True))

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
        proxy_started = False
        for _ in range(50):
            if _port_open(host, port) and _ping(base_url):
                proxy_started = True
                break
            time.sleep(0.1)
        
        if not proxy_started:
            return {"running": False, "base_url": base_url, "message": "Proxy thread started but healthcheck failed."}
        
        # 프록시가 성공적으로 시작된 후 Discovery 서비스에 에이전트 등록
        discovery_message = ""
        if auto_register_discovery and discovery_url and agent_card_payload:
            try:
                log.warning(f"Discovery 서비스 등록 시도: {discovery_url}")
                success, msg = _register_agent_to_discovery(
                    discovery_url=discovery_url,
                    agent_card=agent_card_payload,
                    base_url=base_url
                )
                discovery_message = f" Discovery: {msg}" if msg else ""
                if not success:
                    log.warning(f"Discovery 서비스 등록 실패: {msg}")
            except Exception as e:
                discovery_message = f" Discovery: 등록 중 오류 발생 - {str(e)}"
                log.error(f"Discovery 서비스 등록 중 예외: {e}")
        
        return {
            "running": True, 
            "base_url": base_url, 
            "message": f"Proxy started. {pre_msg}{discovery_message}"
        }
