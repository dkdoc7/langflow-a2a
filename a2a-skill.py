import json
from typing import Any, Dict, List, Optional
import uuid
from langflow.custom.custom_component.component import Component
from langflow.io import MessageTextInput, Output, HandleInput
from langflow.schema.message import Message
from langflow.logging import logger as log


def _parse_list(value: Optional[str]) -> List[str]:
    """
    쉼표 CSV 또는 JSON 배열 문자열을 리스트로 변환.
    예: "tag1, tag2" 또는 ["tag1","tag2"]
    """
    if not value:
        return []
    txt = value.strip()
    # JSON 배열 지원
    if txt.startswith("[") and txt.endswith("]"):
        try:
            arr = json.loads(txt)
            return [str(x) for x in arr]
        except Exception:
            pass
    # CSV
    return [s.strip() for s in txt.split(",") if s.strip()]


class SkillComponent(Component):
    """
    A2A Skill 생성 컴포넌트.
    - 입력: id, name (필수), description, tags, examples(선택)
    - 출력: Message(text=<pretty JSON>)
    """
    display_name: str = "Skill (A2A)"
    description: str = "Build a single A2A skill object."
    icon = "wrench"
    trace_type = "data"
    name = "Skill"

    inputs = [
        # 필수
        StrInput(name="name", display_name="Skill Name", required=True),
        StrInput(name="id", display_name="Skill ID", value=str(uuid.uuid4()),advanced=True, dynamic=True),
        # 선택(Advanced)
        StrInput(name="description", display_name="Description", advanced=True),
        StrInput(
            name="tags",
            display_name="Tags (CSV or JSON Array)",
            advanced=True,
            info='e.g. "currency conversion,currency exchange" or ["currency conversion","currency exchange"]',
        ),
        StrInput(
            name="examples",
            display_name="Examples (CSV or JSON Array)",
            advanced=True,
            info='e.g. "What is USD/GBP?,Convert EUR to JPY" or ["What is USD/GBP?","Convert EUR to JPY"]',
        ),
    ]

    outputs = [
        Output(display_name="Skill (JSON text)", name="skill_json", method="build_skill"),
    ]

    def initialize_data(self) -> None:
        """새 노드(붙여넣기 포함)마다 고유 UUID 부여"""
        self._attributes["id"]= str(uuid.uuid4())
        self.status=self._attributes["id"]
        log.info(f"[SkillCard] initialize_data ran: {self.id}")

    async def build_skill(self) -> Message:
        self.id=str(uuid.uuid4())
        self._attributes["id"]= str(uuid.uuid4())
        skill: Dict[str, Any] = {
            "id":  self._attributes.get("id") or "",
            "name": self._attributes.get("name") or "",
            "description": self._attributes.get("description") or "",
            "tags": _parse_list(self._attributes.get("tags")),
            "examples": _parse_list(self._attributes.get("examples")),
        }
        txt = json.dumps(skill, ensure_ascii=False, indent=2)
        self.status = self.id  # 프리뷰
        return Message(text=txt)


