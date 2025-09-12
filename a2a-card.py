import json
from typing import Any, Dict, List, Optional

from langflow.custom.custom_component.component import Component
from langflow.io import (
    MessageTextInput,
    Output,
    BoolInput,
    HandleInput,
    StrInput,
    DropdownInput,
)
from langflow.schema.message import Message


# ---------- 헬퍼 ---------- #
def _parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    txt = value.strip()
    if txt.startswith("[") and txt.endswith("]"):
        try:
            return [str(x) for x in json.loads(txt)]
        except Exception:
            pass
    return [s.strip() for s in txt.split(",") if s.strip()]


def _parse_skills_text(value: Optional[str]) -> List[Dict[str, Any]]:
    if not value:
        return []
    txt = value.strip()
    # JSON 배열?
    if txt.startswith("["):
        try:
            data = json.loads(txt)
            return data if isinstance(data, list) else []
        except Exception:
            return []
    # 단일 JSON 객체?
    if txt.startswith("{") and txt.endswith("}"):
        try:
            data = json.loads(txt)
            return [data] if isinstance(data, dict) else []
        except Exception:
            return []
    # 간소 CSV 파싱
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if lines and ("=" in lines[0] and ":" in lines[0]):
        lines = lines[1:]
    skills: List[Dict[str, Any]] = []
    for ln in lines:
        segs = ln.split("=")
        if len(segs) < 3:
            continue
        sid, sname = segs[0].strip(), segs[1].strip()
        rest = "=".join(segs[2:]).strip()
        tags, examples = [], []
        if ":" in rest:
            desc_part, tail = rest.split(":", 1)
            sdesc = desc_part.strip().strip('"')
            tail = tail.strip()
            if ":" in tail:
                tags_raw, ex_raw = tail.split(":", 1)
                tags = [t.strip() for t in tags_raw.strip('"').split("|") if t.strip()]
                examples = [e.strip() for e in ex_raw.strip('"').split("|") if e.strip()]
            else:
                tags = [t.strip() for t in tail.strip('"').split("|") if t.strip()]
        else:
            sdesc = rest.strip().strip('"')
        skills.append(
            {"id": sid, "name": sname, "description": sdesc, "tags": tags, "examples": examples}
        )
    return skills


def _parse_skills_handle(msg_list: List[Message]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for msg in msg_list or []:
        if not isinstance(msg, Message) or not getattr(msg, "text", None):
            continue
        try:
            data = json.loads(msg.text)
            if isinstance(data, list):
                results.extend([d for d in data if isinstance(d, dict)])
            elif isinstance(data, dict):
                results.append(data)
        except Exception:
            continue
    return results


# ---------- 컴포넌트 ---------- #
class AgentCardComponent(Component):
    display_name: str = "Agent Card (A2A)"
    description: str = "Build an A2A-compatible Agent Card JSON."
    icon = "id-card"
    trace_type = "data"
    name = "AgentCard"

    inputs = [
        # 필수
        StrInput(name="name", display_name="Name", required=False),
        StrInput(name="id", display_name="ID", required=True),
        StrInput(
            name="url", display_name="URL", required=True, info="Base URL or entrypoint of the agent.",real_time_refresh=True
        ),
        # 선택(Advanced)
        StrInput(name="description", display_name="Description", advanced=True),
        StrInput(name="version", display_name="Version", value="1.0.0", advanced=True),
        BoolInput(name="cap_streaming", display_name="Capabilities.Streaming", value=True, advanced=True),
        BoolInput(name="cap_push", display_name="Capabilities.PushNotifications", value=False, advanced=True),
        BoolInput(name="cap_state", display_name="Capabilities.StateTransitionHistory", value=False, advanced=True),
        DropdownInput(
            name="preferred_transport",
            display_name="Preferred Transport",
            advanced=True,
            options=["JSONRPC","GRPC","HTTP+JSON"],
            value="JSONRPC"
        ),
        MessageTextInput(
            name="default_input_modes",
            display_name="Default Input Modes (CSV or JSON Array)",
            value="text,text/plain",
            advanced=True,
        ),
        MessageTextInput(
            name="default_output_modes",
            display_name="Default Output Modes (CSV or JSON Array)",
            value="text,text/plain",
            advanced=True,
        ),
        HandleInput(
            name="skills_handle",
            display_name="Skills (connect multiple Skill components)",
            input_types=["Message"],
            is_list=True,
            advanced=False,  # UI에 보이되 빈 상태여도 OK
        ),
        # extra
        MessageTextInput(
            name="extra",
            display_name="Extra Fields (JSON, optional)",
            advanced=True,
            info='Example: {"owner":"team-a","license":"MIT"}',
        ),
    ]

    outputs = [
        Output(display_name="Agent Card (JSON text)", name="card_json", method="build_card_message"),
    ]
    
    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None) -> dict:
        if field_name == "url":
            self.build_card_message()
        return build_config
        

    # ---------- 내부 ---------- #
    def _merge_skills(self) -> List[Dict[str, Any]]:
        # 텍스트 기반
        skills_text = _parse_skills_text(self._attributes.get("skills"))
        # HandleInput 기반
        skills_handle = _parse_skills_handle(self._attributes.get("skills_handle") or [])
        combined = skills_text + skills_handle

        # id 기준 dedup (뒤에 온 것 우선)
        dedup: Dict[str, Dict[str, Any]] = {}
        for s in combined:
            sid = str(s.get("id", "")).strip()
            key = sid if sid else f"__idx_{len(dedup)}"
            dedup[key] = s
        return list(dedup.values())

    def _compose_card(self) -> Dict[str, Any]:
        card: Dict[str, Any] = {
            "name": self._attributes.get("name") or "",
            "id" : self._attributes.get("id") or "",
            "description": self._attributes.get("description") or "",
            "url": self._attributes.get("url") or "",
            "version": self._attributes.get("version") or "1.0.0",
            "capabilities": {
                "streaming": bool(self._attributes.get("cap_streaming", True)),
                "pushNotifications": bool(self._attributes.get("cap_push", False)),
                "stateTransitionHistory": bool(self._attributes.get("cap_state", False)),
            },
            "defaultInputModes": _parse_list(self._attributes.get("default_input_modes")),
            "defaultOutputModes": _parse_list(self._attributes.get("default_output_modes")),
            "skills": self._merge_skills(),
        }

        # extra 병합
        extra_raw = self._attributes.get("extra")
        if extra_raw:
            try:
                extra_obj = json.loads(extra_raw)
                if isinstance(extra_obj, dict):
                    for k, v in extra_obj.items():
                        if k not in card:
                            card[k] = v
                        elif isinstance(card[k], dict) and isinstance(v, dict):
                            card[k] = {**card[k], **v}
            except Exception:
                pass
        return card

    async def build_card_message(self) -> Message:
        card = self._compose_card()
        txt = json.dumps(card, ensure_ascii=False, indent=2)
        self.status = f"{len(card['skills'])} skill(s) linked"
        return Message(text=txt)