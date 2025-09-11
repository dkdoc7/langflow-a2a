import json
import asyncio
from typing import Any, Dict, List, Optional
import aiohttp
import re

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS,
    MODEL_PROVIDERS_DICT,
    MODELS_METADATA,
)
from langflow.components.helpers.memory import MemoryComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.utils import update_component_build_config
from langflow.field_typing import Tool
from langflow.io import BoolInput, DropdownInput, HandleInput, IntInput, MultilineInput, Output, StrInput
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_AI
import re


def decide_plan_update(feedback: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    """
    계획 수정 필요성 판단:
    - is_complete가 True면 수정 불필요
    - 개선 제안이 존재하면(그리고 비어있지 않으면) 수정 고려
    - 명시적 오류/누락 시 수정
    """
    if not isinstance(feedback, dict):
        return False

    if feedback.get("is_complete", False):
        return False

    suggestions = feedback.get("improvement_suggestions") or []
    if isinstance(suggestions, list) and len(suggestions) > 0:
        return True

    critique = (feedback.get("critique") or "").lower()
    keywords = ["오류", "누락", "부족", "실패", "fail", "missing", "error"]
    if any(k in critique for k in keywords):
        return True

    return False


def apply_feedback_to_plan(plan: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
    """피드백을 계획에 반영. 보수적으로 execution_plan에 반영 메모/완화책만 추가."""
    try:
        updated = dict(plan)
        updated["updated"] = True

        critique = feedback.get("critique")
        suggestions = feedback.get("improvement_suggestions") or []

        exec_plan = list(updated.get("execution_plan") or [])
        exec_plan.append({
            "phase": "Feedback Incorporation",
            "targets": [t.get("id") for t in (updated.get("work_breakdown") or []) if isinstance(t, dict)],
            "notes": (critique or "")[:500]
        })
        updated["execution_plan"] = exec_plan

        migi = list(updated.get("mitigations_global") or [])
        if suggestions:
            migi.extend(suggestions if isinstance(suggestions, list) else [str(suggestions)])
        updated["mitigations_global"] = migi

        return updated
    except Exception:
        return plan


def create_prompt_from_component(prompt_component, default_system_message: str = "") -> ChatPromptTemplate:
    if not prompt_component:
        logger.warning("prompt_component is None or empty")
        return None
    try:
        # 다양한 속성에서 템플릿 내용 추출
        template_content = None
        if hasattr(prompt_component, "template"):
            template_content = prompt_component.template
        elif hasattr(prompt_component, "content"):
            template_content = prompt_component.content
        elif hasattr(prompt_component, "text"):
            template_content = prompt_component.text
        else:
            # Message 객체의 경우 직접 문자열로 변환 시도
            template_content = str(prompt_component)
        
        if not template_content or not template_content.strip():
            logger.warning("Template content is empty")
            return None
                    
        if "{chat_history}" in str(template_content):
            msgs = []
            if default_system_message:
                msgs.append(("system", default_system_message))
            msgs.extend([MessagesPlaceholder(variable_name="chat_history"), ("human", template_content)])
            return ChatPromptTemplate.from_messages(msgs)
        msgs = [("human", template_content)]
        if default_system_message:
            msgs = [("system", default_system_message)] + msgs
        return ChatPromptTemplate.from_messages(msgs)
    except Exception as e:
        logger.error(f"Error creating prompt from component: {e}")
        return None


def decode_unicode_escapes(text: str) -> str:
    """주어진 문자열에 \\uXXXX 형태의 유니코드 이스케이프가 포함되면 사람이 읽을 수 있는 문자로 변환한다."""
    if not isinstance(text, str):
        return text
    # JSON 문자열 리터럴 형태("\uc548...\u...")인 경우 파싱 시도
    try:
        loaded = json.loads(text)
        if isinstance(loaded, str):
            return loaded
    except Exception:
        pass
    # 일반 문자열에서 유니코드 이스케이프 디코딩 시도
    if "\\u" in text:
        try:
            return text.encode("utf-8").decode("unicode_escape")
        except Exception:
            pass
    return text

def find_filed(obj, key):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                return v
            result = find_filed(v, key)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = find_filed(item, key)
            if result:
                return result
    return None


MODEL_PROVIDERS_LIST = ["Anthropic", "Google Generative AI", "Groq", "OpenAI"]


class A2APDCAgentComponent(ToolCallingAgentComponent):
    display_name: str = "A2A PDCA Agent"
    description: str = "Plan-Do-Critic(Act) 사이클을 하나의 컴포넌트에서 오케스트레이션"
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "A2A PDCA Agent"

    # 공통부(맨 위) → 단계별 입력 순서대로 배치
    inputs = [
         HandleInput(
            name="plan_input",
            display_name="Plan Schema (JSON, optional)",
            info="테스트용 계획(JSON)",
            advanced=False,
            input_types=["Message"],
            required=False,
        ),
       
        # 공통: LLM/Provider
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            input_types=["LanguageModel"],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST] + [{"icon": "brain"}],
        ),
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],
        StrInput(
            name="a2a_api_base",
            display_name="A2A API Base URL",
            info="A2A Discovery 서비스의 API 기본 URL",
            value="http://localhost:8000",
            advanced=False,
        ),
        StrInput(
            name="input_value",
            display_name="User Input",
            info="사용자 입력 (Message 연결 가능)",
            value="",
            input_types=["Message"],
            advanced=False,
        ),

        BoolInput(
            name="verbose",
            display_name="Verbose",
            value=False,
            advanced=True,
        ),
        IntInput(
            name="a2a_max_iterations",
            display_name="PDCA Max Iterations",
            value=3,
            info="PDCA 사이클 최대 반복 횟수",
            advanced=True,
        ),
        # 1) Plan 단계
        HandleInput(
            name="work_plan_prompt",
            display_name="Work Plan Prompt (필수)",
            info="작업 계획 수립 프롬프트",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),
        HandleInput(
            name="plan_schema",
            display_name="Plan Schema (JSON, optional)",
            info="계획 출력 스키마(JSON)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="시스템 지시사항 (선택)",
            value="You are a helpful assistant.",
            advanced=False,
        ),

        # 2) Do(Delegator) 단계: 별도 입력 없음 (plan/agents를 기반으로 실행)

        # 3) Critic(Act) 단계
        HandleInput(
            name="critic_prompt",
            display_name="Critic Prompt (필수)",
            info="실행 결과 평가 프롬프트",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),
        HandleInput(
            name="tools",
            display_name="Tools",
            input_types=["Tool"],
            is_list=True,
            required=False,
        ),


    ]

    outputs = [
        # 종합 결과
        Output(name="response", display_name="Response", group_outputs=True, method="message_response"),
        # 단계별 결과
        Output(name="plan", display_name="Plan", group_outputs=True, method="plan_output"),
        Output(name="execution", display_name="Execution", group_outputs=True, method="execution_output"),
        Output(name="critique", display_name="Critique", group_outputs=True, method="critique_output"),
        Output(name="updated_plan", display_name="Updated Plan", group_outputs=True, method="updated_plan_output"),
        Output(name="done", display_name="Done", group_outputs=True, method="done_output"),
    ]

    async def message_response(self) -> Message:
        logger.info(f"-- PDC : Message response called")
        try:
            await self._initialize_llm()
            # 모든 input 포트 점검 (런타임 전용)
            self._validate_required_ports()

            available_agents = await self.get_available_agents()
            # 디버깅 용. plan_input 포트가 연결된 경우 우선 사용 (Message → JSON 파싱)
            if hasattr(self, "plan_input") and self.plan_input:
                raw = getattr(self.plan_input, "text", None) or getattr(self.plan_input, "content", None) or str(self.plan_input)
                plan = self._parse_json_string_relaxed(raw) if isinstance(raw, str) and raw.strip() else {}
            else:
                plan = await self.create_work_plan(available_agents)
            
            self._last_plan = plan
            self.plan_output()

            # PDCA 외부 루프: Dispatcher → Critic → (조건부 Plan 업데이트)
            existing_results: Dict[str, Any] = {}
            critique_results: Dict[str, Any] = {}
            iterations = 0
            total_tasks = len(plan.get("work_breakdown", []))
            is_done = False
            last_task_id = ""

            while iterations < int(getattr(self, "a2a_max_iterations", 1) or 1):
                task_id, task_outputs, new_results = await self.dispatcher_phase(plan, available_agents, existing_results)
                last_task_id = task_id
                if not task_id:
                    # 실행할 태스크 없음 → 모든 작업 완료로 간주
                    is_done = True
                    critique = {"is_complete": True}
                    break
                
                result_data=new_results.get(task_id).get('result')

                new_result = {
                    task_id: {
                        "status": new_results.get(task_id).get("status"), 
                        "outputs": result_data,
                    }
                }
                # 결과 합치기
                existing_results[task_id] = new_result
                self._last_execution = existing_results
                self.execution_output()

                # 비평 단계
                critique = await self.critic_phase(task_id, existing_results, plan)
                critique_results[task_id] = critique
                self._last_critique = critique_results
                self.critique_output()

                logger.info(f"-- PDC : Critic phase has been completed.")

                # 조건부 계획 업데이트
                if decide_plan_update(critique, plan):
                    plan = apply_feedback_to_plan(plan, critique)
                    total_tasks = len(plan.get("work_breakdown", []))
                    self._last_updated_plan=plan
                    self.updated_plan_output()

                # 완료 조건: critic 신호 또는 모든 태스크 완료
                is_done = bool(critique.get("is_complete", False)) or (len(existing_results) >= total_tasks)

                if is_done:
                    self._last_done = is_done
                    self.done_output()
                    break
                iterations += 1

            summary = {
                "status": "completed" if is_done else "in_progress",
                "executed_task": last_task_id,
                "iterations": iterations + (1 if last_task_id else 0),
            }
            return Message(text=json.dumps(summary, ensure_ascii=False, indent=2), sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)
        except Exception as e:
            logger.error(f"A2A PDCA Agent error: {e}")
            return Message(text=f"A2A PDCA Agent error: {e}")

    # ------------ 개별 출력 ------------
    def plan_output(self) -> Message:
        txt = json.dumps(getattr(self, "_last_plan", {}) or {}, ensure_ascii=False, indent=2)
        return Message(text=txt, sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)

    def execution_output(self) -> Message:
        txt = json.dumps(getattr(self, "_last_execution", {}) or {}, ensure_ascii=False, indent=2)
        return Message(text=txt, sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)

    def critique_output(self) -> Message:
        txt = json.dumps(getattr(self, "_last_critique", {}) or {}, ensure_ascii=False, indent=2)
        return Message(text=txt, sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)

    def updated_plan_output(self) -> Message:
        txt = json.dumps(getattr(self, "_last_updated_plan", {}) or {}, ensure_ascii=False, indent=2)
        return Message(text=txt, sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)

    def done_output(self) -> Message:
        val = "true" if bool(getattr(self, "_last_done", False)) else "false"
        return Message(text=val, sender=MESSAGE_SENDER_AI, sender_name=MESSAGE_SENDER_NAME_AI)

    # ------------ 내부 헬퍼 (a2a-agent.py에서 차용/경량화) ------------
    def _validate_required_ports(self) -> None:
        """모든 필수 input 포트 점검"""
        # LLM 필수 확인
        if not hasattr(self, "llm") or self.llm is None:
            raise ValueError("LLM이 연결되지 않았습니다. Language Model 컴포넌트를 연결해주세요.")
        
        # input_value 필수 확인
        if not hasattr(self, "input_value") or not self.input_value:
            raise ValueError("입력값이 제공되지 않았습니다. Input Value를 연결해주세요.")
        
        user_task = self.input_value.content if hasattr(self.input_value, "content") else str(self.input_value)
        user_task = decode_unicode_escapes(user_task)
        if not user_task or not user_task.strip():
            raise ValueError("입력값이 비어있습니다. 유효한 작업 내용을 입력해주세요.")
        
        # Work Plan Prompt 필수 확인
        if not hasattr(self, "work_plan_prompt") or not self.work_plan_prompt:
            raise ValueError("Work Plan Prompt가 연결되지 않았습니다. Prompt Template 컴포넌트를 연결해주세요.")
        
        # Critic Prompt 필수 확인
        if not hasattr(self, "critic_prompt") or not self.critic_prompt:
            raise ValueError("Critic Prompt가 연결되지 않았습니다. Prompt Template 컴포넌트를 연결해주세요.")
        
        if not hasattr(self, "plan_schema") or not self.plan_schema:
            raise ValueError("작업계획 Schema가 제공되지 않았습니다. 작업계획 Schema를 연결해주세요.")
        

    async def _initialize_llm(self):
        try:
            if not hasattr(self, "chat_history"):
                self.chat_history = []
            if hasattr(self, "llm") and self.llm is not None and not isinstance(self.llm, str):
                return
            if hasattr(self, "agent_llm") and self.agent_llm is not None:
                if isinstance(self.agent_llm, str):
                    # 문자열인 경우 실제 LLM 객체로 변환
                    self.llm, _ = self.get_llm()
                else:
                    self.llm = self.agent_llm
            else:
                self.llm = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    def _parse_json_string_relaxed(self, text: str) -> Dict[str, Any]:
        """약간의 따옴표 오류를 교정하며 JSON 파싱을 시도.
        예: ""안녕하세요"" → "안녕하세요"
        """
        # 1차 시도: 그대로 파싱
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2차 시도: 연속 쌍따옴표로 감싼 구절을 단일로 교정
        try:
            fixed = re.sub(r'""([^"\\\n]+)""', r'"\1"', text)
            return json.loads(fixed)
        except Exception:
            pass
        # 더 이상 교정 불가
        raise ValueError("유효한 JSON 형식이 아닙니다.")

    def get_llm(self):
        if not isinstance(self.agent_llm, str):
            return self.agent_llm, None

        try:
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if not provider_info:
                msg = f"Invalid model provider: {self.agent_llm}"
                raise ValueError(msg)

            component_class = provider_info.get("component_class")
            display_name = component_class.display_name
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix", "")

            return self._build_llm_model(component_class, inputs, prefix), display_name

        except Exception as e:
            logger.error(f"Error building {self.agent_llm} language model: {e!s}")
            msg = f"Failed to initialize language model: {e!s}"
            raise ValueError(msg) from e

    def _build_llm_model(self, component, inputs, prefix=""):
        model_kwargs = {}
        for input_ in inputs:
            if hasattr(self, f"{prefix}{input_.name}"):
                value = getattr(self, f"{prefix}{input_.name}")
                if value is not None:  # None 값 제외
                    model_kwargs[input_.name] = value
        
        try:
            if model_kwargs:
                return component.set(**model_kwargs).build_model()
            else:
                return component.build_model()
        except Exception as e:
            logger.warning(f"Component.set() 실패, 기본 설정으로 시도: {e}")
            try:
                return component.build_model()
            except Exception as e2:
                logger.error(f"Component.build_model()도 실패: {e2}")
                raise

    async def get_available_agents(self) -> List[Dict[str, Any]]:
        try:
            api_url = f"{self.a2a_api_base}/agents?status=active"
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if isinstance(data, dict) and "agents" in data:
                            logger.info(f"A2A Discovery : Found {len(data['agents'])} available agents.")
                            return data["agents"]
                        return []
                    return []
        except Exception:
            return []

    async def create_work_plan(self, available_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        planning_prompt = create_prompt_from_component(self.work_plan_prompt, default_system_message=self.system_prompt)
        if planning_prompt is None:
            raise ValueError("Work Plan Prompt 변환에 실패했습니다. 올바른 프롬프트 템플릿을 연결해주세요.")
        
        # output_schema 입력 포트 처리 (Message -> JSON)
        output_schema = None
        raw = getattr(self.plan_schema, "text", None) or getattr(self.plan_schema, "content", None) or str(self.plan_schema)
        if isinstance(raw, str) and raw.strip():
            output_schema = self._parse_json_string_relaxed(raw)

        user_task = self.input_value.content if hasattr(self.input_value, "content") else str(self.input_value)
        user_task = decode_unicode_escapes(user_task)
        chain = planning_prompt | self.llm | JsonOutputParser()

        logger.info(f"-- PDC : Planning Started...")
        result = await chain.ainvoke({
            "user_task": user_task,
            "available_agents": json.dumps(available_agents, ensure_ascii=False, indent=2),
            "output_schema": json.dumps(output_schema, ensure_ascii=False, indent=2),
            "chat_history": getattr(self, "chat_history", []),
        })
        logger.info(f"-- PDC : Planning has been completed with {len(result['work_breakdown'])} tasks")
        # self.plan 출력을 업데이트
        self.plan = result
        return result

    async def dispatcher_phase(self, plan: Dict[str, Any], available_agents: List[Dict[str, Any]], existing_results: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        tasks = plan.get("work_breakdown", [])
        task_id = ""
        task_outputs: Dict[str, Any] = {}
        new_results: Dict[str, Any] = {}
        for task in tasks:
            tid = task.get("id")
            task_outputs = task.get("outputs", [])
            if tid in (existing_results or {}):
                continue
            task_id = tid
            agent_id = task.get("agent").get("id")
            try:
                if agent_id and agent_id != "self":
                    logger.info(f"-- PDC : Delegating task to agent {agent_id}")
                    res = await self.delegate_task_to_agent(task, agent_id, available_agents)
                    #res->result->parts[0]를 추출. 실패일 경우 기본 응답 생성
                    res = res.get("result", {}).get("parts", [{}])[0].get("text", "")
                    new_results[task_id] = {"status": "completed","execution_method": "agent", "result": res, "agent": agent_id}
                else:
                    logger.info(f"-- PDC : Executing task directly")
                    res = await self.execute_task_directly(task, existing_results)
                    res = res.get("result", {})
                    new_results[task_id] = {"status": "completed","execution_method": "self", "result": res, "agent": agent_id}
            except Exception as e:
                new_results[task_id] = {"status": "failed", "error": str(e), "agent": agent_id}
            break
        return task_id, task_outputs, new_results

    async def critic_phase(self, task_id: str, execution_results: Dict[str, Any], current_plan: Dict[str, Any]) -> Dict[str, Any]:

        # 계획 프롬프트 처리 방식 참고: 명시적인 시스템/휴먼 메시지 구성 및 JsonOutputParser 사용
        # 1) 출력 스키마 정의 (LLM이 정확히 이 구조로만 반환하도록 강제)
        output_schema = {
            "type": "object",
            "properties": {
                "critique": {"type": "string"},
                "is_complete": {"type": "boolean"},
                "output_values": {"type": "array", "items": {"type": "object"}},
                "improvement_suggestions": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["critique", "is_complete", "output_values"],
            "additionalProperties": False
        }

        # 2) 안전 직렬화 헬퍼
        def safe(obj: Any) -> str:
            try:
                return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
            except Exception:
                return str(obj)

        # 3) 시스템/휴먼 메시지 기반 프롬프트 구성
        try:
            from langchain_core.prompts import ChatPromptTemplate

            system_prompt_text = getattr(self, "system_prompt", None) or "You are a strict but fair critic. Return JSON only."
            # critic_prompt 텍스트 추출 (plan 처리와 유사하게 다양한 속성에서 추출)
            if hasattr(self.critic_prompt, "template"):
                critic_text = self.critic_prompt.template
            elif hasattr(self.critic_prompt, "content"):
                critic_text = self.critic_prompt.content
            elif hasattr(self.critic_prompt, "text"):
                critic_text = self.critic_prompt.text
            else:
                critic_text = str(self.critic_prompt)

            if not critic_text or not str(critic_text).strip():
                raise ValueError("Critic Prompt 변환에 실패했습니다. 올바른 프롬프트 템플릿을 연결해주세요.")
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_prompt}\n\n{critic_prompt}\n\nReturn a SINGLE JSON object that strictly matches the given output_schema. No markdown fences or extra text."),
                ("human", "Execution Results to Evaluate:\n{execution_results}\n\nCurrent Plan:\n{current_plan}\n\nOutput Schema:\n{output_schema}")
            ])

            chain = prompt | self.llm | JsonOutputParser()
            #current_plan에서 execution_plan을 제외하고 추출 (이후 critic 작업에 영향을 미치는 요소들 제거. 중요함!)
            _cp = current_plan
            cp = {
                "goal": _cp.get("goal", ""),
                "assumptions": _cp.get("assumptions", []),
                "critical_path": _cp.get("critical_path", []),
                "risk_global": _cp.get("risk_global", []),
                "work_breakdown": _cp.get("work_breakdown", []),
                }

            result = await chain.ainvoke({
                "system_prompt": system_prompt_text,
                "critic_prompt": critic_text,
                "execution_results": safe(execution_results),
                "current_plan": safe(cp),
                "output_schema": json.dumps(output_schema, ensure_ascii=False, indent=2),
            })
        except Exception as e:
            logger.error(f"JSON 파싱 실패, 기본 응답으로 대체: {e}")
            # JSON 파싱 실패 시 기본 응답 생성
            result = {
                "is_complete": False,
                "critique": "JSON 파싱 오류로 인한 기본 응답",
                "improvement_suggestions": [],
                "output_values": []
            }

        # 4) 누락 필드 보정 (plan 처리와 동일한 보수적 보정)
        if "is_complete" not in result:
            total = len(current_plan.get("work_breakdown", []))
            done = 1 if task_id else 0
            result["is_complete"] = (done >= total)
        if "output_values" not in result:
            result["output_values"] = []
        return result

    # ---- 이하 메서드들은 a2a-agent.py 구현에서 발췌/경량화 ----
    async def execute_task_directly(self, task: Dict[str, Any], existing_results: Dict[str, Any] = None) -> Dict[str, Any]:
        title = task.get("title", "일반 작업")
        #user_input = self.input_value.content if hasattr(self.input_value, "content") else str(self.input_value)
        if hasattr(self, "llm") and self.llm is not None:
            try:
                task_description = task.get('description', '')
                task_inputs = task.get('inputs', [])
                prompt = f"""다음 작업을 수행해주세요:\n\n작업: {task_description}\n입력: {task_inputs}\n\n이전작업 결과: {existing_results}\n\n결과를 간결히 반환하세요."""
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                content = response.content if hasattr(response, "content") else str(response)
                return {"task_id": task.get("id"), "execution_method": "llm_general", "status": "completed", "result": content}
                
            except Exception:
                pass
        return {"task_id": task.get("id"), "execution_method": "basic_general", "status": "completed", "result": f"Task '{title}' processed."}

    async def delegate_task_to_agent(self, task: Dict[str, Any], agent_id: str, available_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        target = next((a for a in available_agents if a.get("id") == agent_id), None)
        if not target:
            raise ValueError(f"Agent '{agent_id}' not found")
        agent_url = target.get("url") or target.get("endpoint")
        # URL 타입 보정 및 검증'
        if agent_url is None:
            raise ValueError("Agent URL not found")
        if not isinstance(agent_url, str):
            agent_url = str(agent_url)
        if not agent_url:
            raise ValueError("Agent URL not found")
        if isinstance(agent_url, str) and "0.0.0.0" in agent_url:
            agent_url = agent_url.replace("0.0.0.0", "127.0.0.1")
        verbose_prefix = f"**반드시, 다음 json 으로 변환하여 반환해주세요.**\n반환형식\n{{'output 요소':'output 값', ... }}\n\n" if not self.verbose else ""
        payload = {
            "jsonrpc": "2.0",
            "id": task["id"],
            "method": "invoke",
            "params": {
                "message": {
                    "message_id": task["id"],
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": f"*응답은 주어진 task의 outputs에 해당하는 결과만 생성할것!*\n{verbose_prefix}Task: {task['title']}\nDescription: {task.get('description', '')}\nInputs: {', '.join(task.get('inputs', []))}\nExpected Outputs: {', '.join(task.get('outputs', []))}"
                        }
                    ]
                },
                "stream": False  # 스트리밍 비활성화
            }
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        async with aiohttp.ClientSession() as session:
            async with session.post(agent_url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data
                raise ValueError(f"Agent returned status {resp.status}")


