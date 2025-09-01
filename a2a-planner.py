import json
import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from uuid import uuid4
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import StructuredTool

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.base.agents.events import ExceptionWithMessageError
from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_DYNAMIC_UPDATE_FIELDS,
    MODEL_PROVIDERS,
    MODEL_PROVIDERS_DICT,
    MODELS_METADATA,
)
from langflow.base.models.model_utils import get_model_name
from langflow.components.helpers.current_date import CurrentDateComponent
from langflow.components.helpers.memory import MemoryComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.custom_component.component import _get_component_toolkit
from langflow.custom.utils import update_component_build_config
from langflow.field_typing import Tool
from langflow.io import BoolInput, DropdownInput, HandleInput, IntInput, MultilineInput, Output, StrInput
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_AI


def create_prompt_from_component(prompt_component, default_system_message: str = "") -> ChatPromptTemplate:
    """
    프롬프트 컴포넌트에서 ChatPromptTemplate을 생성하는 헬퍼 함수
    
    Args:
        prompt_component: Langflow Prompt Component에서 출력된 Message 객체
        default_system_message: 기본 시스템 메시지
    
    Returns:
        ChatPromptTemplate: LangChain 프롬프트 템플릿
    """
    if not prompt_component:
        logger.warning("prompt_component is None or empty")
        return None
        
    try:
        # Message 객체에서 template 추출
        if hasattr(prompt_component, 'template'):
            template_content = prompt_component.template
        elif hasattr(prompt_component, 'text'):
            template_content = prompt_component.text
        else:
            logger.warning("Prompt component format not recognized")
            return None
        
        # ChatPromptTemplate 생성 (변수는 런타임에 주입)
        # chat_history 변수가 포함된 경우 MessagesPlaceholder 추가
        if "{chat_history}" in template_content:
            if default_system_message:
                return ChatPromptTemplate.from_messages([
                    ("system", default_system_message),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", template_content)
                ])
            else:
                return ChatPromptTemplate.from_messages([
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", template_content)
                ])
        else:
            if default_system_message:
                return ChatPromptTemplate.from_messages([
                    ("system", default_system_message),
                    ("human", template_content)
                ])
            else:
                return ChatPromptTemplate.from_messages([
                    ("human", template_content)
                ])
            
    except Exception as e:
        logger.error(f"Error creating prompt from component: {e}")
        return None


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


MODEL_PROVIDERS_LIST = ["Anthropic", "Google Generative AI", "Groq", "OpenAI"]


class A2APlannerComponent(ToolCallingAgentComponent):
    display_name: str = "A2A Planner"
    description: str = "A2A Discovery 서비스와 연동하여 에이전트 목록을 조회하고 작업 계획을 수립하는 플래너"
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "A2A Planner"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        # LLM 연결을 위한 입력 필드
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            input_types=["LanguageModel"],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST] + [{"icon": "brain"}],
        ),
        # OpenAI 모델 설정 입력들
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],

        StrInput(
            name="a2a_api_base",
            display_name="A2A API Base URL",
            info="A2A Discovery 서비스의 API 기본 URL (예: http://localhost:8000, https://api.example.com)",
            value="http://localhost:8000",
            advanced=False,
        ),

        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
        # ChatInput 연결을 위한 입력 필드
        StrInput(
            name="input_value",
            display_name="Input Value",
            info="사용자 입력 메시지 또는 ChatInput에서 연결된 값",
            value="",
            advanced=False,
            input_types=["Message"],
        ),
        # Agent Description 입력 필드
        StrInput(
            name="agent_description",
            display_name="Agent Description",
            info="에이전트에 대한 설명",
            value="A helpful assistant with access to the following tools:",
            advanced=True,
        ),
        # 기본 에이전트 입력들 (max_iterations 제외)
        BoolInput(
            name="handle_parsing_errors",
            display_name="Handle Parsing Errors",
            value=True,
            info="Whether to handle parsing errors.",
            advanced=True,
        ),
        BoolInput(
            name="verbose",
            display_name="Verbose",
            value=False,
            info="Whether to run in verbose mode.",
            advanced=True,
        ),
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
        # Prompt Component 입력 필드들 (필수)
        HandleInput(
            name="system_prompt",
            display_name="System Prompt (필수)",
            info="작업 계획 수립을 위한 시스템 프롬프트 (Prompt Template 컴포넌트 연결 필수)",
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
            info="These are the tools that the agent can use to help with tasks.",
        ),
    ]
    outputs = [Output(name="response", display_name="Response", method="message_response")]

    async def message_response(self) -> Message:
        """A2A Planner의 메인 응답 메서드"""
        logger.info("message_response called - starting A2A Planner execution")

        try:
            # LLM 초기화
            await self._initialize_llm()
            
            # 실제 A2A Planner 로직 실행
            result = await self.run_a2a_planner()
            return result
        except Exception as e:
            logger.error(f"A2A Planner execution error: {e}")
            return Message(
                text=f"A2A Planner 실행 중 오류가 발생했습니다: {str(e)}",
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI
            )

    async def _initialize_llm(self):
        """LLM 및 필수 속성 초기화"""
        try:
            # chat_history 초기화
            if not hasattr(self, 'chat_history'):
                self.chat_history = []
                logger.info("Initialized empty chat_history")
            
            if hasattr(self, 'llm') and self.llm is not None and not isinstance(self.llm, str):
                logger.info("LLM already initialized")
                return
                
            # agent_llm이 설정되어 있는지 확인하고 실제 LLM 객체로 변환
            if hasattr(self, 'agent_llm') and self.agent_llm is not None:
                logger.info(f"Initializing LLM with agent_llm: {type(self.agent_llm)}")
                
                # agent_llm이 문자열인 경우 실제 LLM 객체로 변환
                if isinstance(self.agent_llm, str):
                    logger.info("Converting string agent_llm to actual LLM object")
                    llm_model, display_name = self.get_llm()
                    self.llm = llm_model
                else:
                    # 이미 LLM 객체인 경우
                    self.llm = self.agent_llm
            else:
                logger.warning("No agent_llm provided, LLM will not be available")
                self.llm = None
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    async def run_a2a_planner(self) -> Message:
        """A2A 플래너의 메인 실행 로직"""
        try:
            # 1. A2A Discovery 서비스에서 사용 가능한 에이전트 목록 조회
            available_agents = await self.get_available_agents()

            # 2. 작업 분석 및 계획 수립
            work_plan = await self.create_work_plan(available_agents)

            # 3. 결과 포맷팅 및 반환
            result_text = self.format_planner_result(available_agents, work_plan)
            
            return Message(
                text=result_text,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI
            )

        except Exception as e:
            logger.error(f"Error in A2A planner execution: {e}")
            return Message(
                text=f"A2A 플래너 실행 중 오류가 발생했습니다: {str(e)}",
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI
            )

    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """A2A Discovery 서비스에서 사용 가능한 에이전트 목록을 조회"""
        try:
            api_url = f"{self.a2a_api_base}/agents?status=active"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        response_data = await response.json()
                       
                        # A2A Discovery 서비스는 AgentListResponse 형태로 응답: {"agents": [...]}
                        if isinstance(response_data, dict) and 'agents' in response_data:
                            agents = response_data['agents']
                            logger.info(f"A2A Discovery : Found {len(agents)} available agents.")
                            return agents
                        else:
                            logger.warning(f"A2A Discovery : Unexpected response format from A2A Discovery service: {type(response_data)}")
                            return []
                    else:
                        logger.warning(f"A2A Discovery : Failed to get agents from A2A Discovery service: {response.status}")
                        return []
        except Exception as e:
            logger.warning(f"A2A Discovery : Error connecting to A2A Discovery service: {e}")
            return []

    async def create_work_plan(self, available_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """사용자 작업을 분석하고 작업 분할 계획을 수립"""
        
        # System Prompt 필수 연결 확인
        if not hasattr(self, 'system_prompt') or not self.system_prompt:
            msg = "System Prompt가 연결되지 않았습니다. Prompt Template 컴포넌트를 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)
                
        planning_prompt = create_prompt_from_component(
            self.system_prompt,
            default_system_message=""
        )
        
        if planning_prompt is None:
            msg = "System Prompt 변환에 실패했습니다. 올바른 프롬프트 템플릿을 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)

        # OUTPUT_SCHEMA 정의
        output_schema = {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "전체 목표"},
                "assumptions": {"type": "array", "items": {"type": "string"}, "description": "가정사항들"},
                "updated": {"type": "boolean", "description": "계획이 업데이트되었는지 여부"},
                "work_breakdown": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "태스크 ID (예: T1, T2)"},
                            "title": {"type": "string", "description": "태스크 제목"},
                            "agent": {"type": "string", "description": "담당 에이전트 ID 또는 이름"},
                            "skills": {"type": "array", "items": {"type": "string"}, "description": "필요한 스킬들"},
                            "inputs": {"type": "array", "items": {"type": "string"}, "description": "입력 항목들"},
                            "outputs": {"type": "array", "items": {"type": "string"}, "description": "출력 항목들"},
                            "dod": {"type": "array", "items": {"type": "string"}, "description": "완료 기준(Definition of Done)"},
                            "est_hours": {"type": "number", "description": "추정 소요 시간(시간 단위)"},
                            "dependencies": {"type": "array", "items": {"type": "string"}, "description": "의존성 태스크 ID들"},
                            "parallelizable": {"type": "boolean", "description": "병렬 실행 가능 여부"},
                            "risk": {"type": "array", "items": {"type": "string"}, "description": "리스크 요소들"},
                            "mitigation": {"type": "array", "items": {"type": "string"}, "description": "리스크 완화 방안들"},
                            "bus": {
                                "type": "object",
                                "properties": {
                                    "use_bus": {"type": "boolean"},
                                    "topics": {"type": "array", "items": {"type": "string"}},
                                    "role": {"type": "string", "enum": ["producer", "consumer", "both"]},
                                    "message_schema": {"type": "object"},
                                    "qos": {"type": "object"},
                                    "contracts": {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    }
                },
                "critical_path": {"type": "array", "items": {"type": "string"}, "description": "크리티컬 패스 태스크 ID들"},
                "execution_plan": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "phase": {"type": "string", "description": "실행 단계"},
                            "targets": {"type": "array", "items": {"type": "string"}, "description": "해당 단계의 태스크 ID들"},
                            "notes": {"type": "string", "description": "단계별 참고사항"}
                        }
                    }
                },
                "risks_global": {"type": "array", "items": {"type": "string"}, "description": "전체 프로젝트 리스크"},
                "mitigations_global": {"type": "array", "items": {"type": "string"}, "description": "전체 프로젝트 리스크 완화 방안"}
            }
        }

        # LangChain이 이해할 수 있는 형식으로 메시지 변환
        user_task = self.input_value.content if hasattr(self.input_value, 'content') else str(self.input_value)
        self.user_task = user_task  # 인스턴스 변수로 저장
        
        # LLM이 있는지 확인하고 없으면 예외 발생
        if not hasattr(self, 'llm') or self.llm is None:
            msg = "LLM이 연결되지 않았습니다. Language Model 컴포넌트를 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)

        # LLM을 사용하여 작업 계획 수립
        chain = planning_prompt | self.llm | JsonOutputParser()
        
        try:
            
            # chat_history를 LangChain 형식으로 변환
            langchain_chat_history = []
            if self.chat_history:
                for msg in self.chat_history:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if msg.type == "human":
                            langchain_chat_history.append(HumanMessage(content=msg.content))
                        elif msg.type == "ai":
                            langchain_chat_history.append(AIMessage(content=msg.content))
                        else:
                            langchain_chat_history.append(HumanMessage(content=str(msg.content)))

            result = await chain.ainvoke({
                "user_task": user_task,
                "available_agents": json.dumps(available_agents, ensure_ascii=False, indent=2),
                "output_schema": json.dumps(output_schema, ensure_ascii=False, indent=2),
                "chat_history": langchain_chat_history
            })
            
            logger.info(f"Work plan created with {len(result.get('work_breakdown', []))} tasks")

            # 수립된 작업 제목들 출력
            self._log_work_breakdown(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating work plan: {e}")
            raise

    def get_default_work_plan(self, user_task: str, available_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """LLM이 없을 때 사용할 기본 작업 계획 생성"""
        logger.info("Generating default work plan")

        # 사용자 입력을 분석하여 작업 유형 결정
        task_type = self.analyze_task_type(user_task)
        task_title = self.generate_task_title(user_task, task_type)
        
        # 기본 작업 계획 생성
        work_plan = {
            "goal": f"사용자 요청 처리: {user_task}",
            "assumptions": [
                f"사용자가 {task_type} 작업을 요청했습니다.",
                "내장 기능으로 처리 가능합니다."
            ],
            "work_breakdown": [
                {
                    "id": "T1",
                    "title": task_title,
                    "agent": "self",
                    "skills": [task_type],
                    "inputs": [user_task],
                    "outputs": [f"{task_type} 결과"],
                    "dod": ["작업이 완료됨"],
                    "est_hours": 0.1,
                    "dependencies": [],
                    "parallelizable": False,
                    "risk": ["기본 처리"],
                    "mitigation": ["표준 절차 사용"],
                    "bus": {
                        "use_bus": False,
                        "topics": [],
                        "role": "consumer",
                        "message_schema": {},
                        "qos": {},
                        "contracts": []
                    }
                }
            ],
            "critical_path": ["T1"],
            "execution_plan": [
                {
                    "phase": "Execution",
                    "targets": ["T1"],
                    "notes": f"{task_type} 작업을 직접 실행합니다."
                }
            ],
            "risks_global": ["기본 처리 제한"],
            "mitigations_global": ["필요시 수동 개입"]
        }

        logger.info(f"Default work plan created with {len(work_plan['work_breakdown'])} tasks")
        return work_plan

    def analyze_task_type(self, user_task: str) -> str:
        """사용자 입력을 분석하여 작업 유형을 결정"""
        user_task_lower = user_task.lower()
        
        if any(keyword in user_task_lower for keyword in ["분석", "단어", "문자", "개수", "통계"]):
            return "텍스트 분석"
        elif any(keyword in user_task_lower for keyword in ["질문", "답변", "물어", "알려", "설명", "?", "무엇", "어떻게", "왜"]):
            return "질의응답"
        elif any(keyword in user_task_lower for keyword in ["요약", "정리", "간추", "핵심"]):
            return "요약"
        elif any(keyword in user_task_lower for keyword in ["생성", "만들", "작성", "창작", "쓰기"]):
            return "생성"
        elif any(keyword in user_task_lower for keyword in ["번역", "translate"]):
            return "번역"
        elif any(keyword in user_task_lower for keyword in ["계산", "수학", "곱셈", "나눗셈", "더하기", "빼기"]):
            return "계산"
        else:
            return "일반 작업"

    def generate_task_title(self, user_task: str, task_type: str) -> str:
        """작업 유형에 따른 적절한 제목 생성"""
        if task_type == "텍스트 분석":
            return "텍스트 분석 및 통계 추출"
        elif task_type == "질의응답":
            return "질문 응답 처리"
        elif task_type == "요약":
            return "텍스트 요약 생성"
        elif task_type == "생성":
            return "콘텐츠 생성"
        elif task_type == "번역":
            return "언어 번역"
        elif task_type == "계산":
            return "수학 계산 수행"
        else:
            return "일반 작업 처리"

    def _log_work_breakdown(self, work_plan: Dict[str, Any]) -> None:
        """작업 계획의 작업 목록을 로그로 출력하는 헬퍼 함수"""
        if "work_breakdown" in work_plan:
            tasks = work_plan["work_breakdown"]
            logger.info(f"수립된 작업 목록 ({len(tasks)}개):")
            for i, task in enumerate(tasks, 1):
                title = task.get("title", f"Task {task.get('id', 'Unknown')}")
                outputs = task.get("outputs", [])
                dod = task.get("dod", [])
                logger.info(f"  {i}. {title}\r\n     출력: {', '.join(outputs)}\r\n     완료 기준: {', '.join(dod)}")

    def format_planner_result(self, available_agents: List[Dict[str, Any]], work_plan: Dict[str, Any]) -> str:
        """플래너 결과를 사용자 친화적인 형태로 포맷팅"""
        
        try:
            logger.info("Starting format_planner_result")
            
            # 헤더
            result_text = "🎯 **A2A 플래너 실행 결과**\n\n"
            
            # 사용 가능한 에이전트 목록
            result_text += f"🤖 **사용 가능한 에이전트** ({len(available_agents)}개)\n"
            if available_agents:
                for i, agent in enumerate(available_agents, 1):
                    agent_id = agent.get("id", "Unknown")
                    agent_name = agent.get("name", agent_id)
                    agent_url = agent.get("url", agent.get("endpoint", "N/A"))
                    result_text += f"  {i}. **{agent_name}** (ID: {agent_id})\n"
                    result_text += f"     URL: {agent_url}\n"
            else:
                result_text += "  사용 가능한 에이전트가 없습니다.\n"
            
            result_text += "\n"
            
            # 작업 계획
            result_text += "📋 **작업 계획**\n"
            result_text += f"• 목표: {work_plan.get('goal', 'N/A')}\n"
            
            # 가정사항
            assumptions = work_plan.get('assumptions', [])
            if assumptions:
                result_text += f"• 가정사항:\n"
                for assumption in assumptions:
                    result_text += f"  - {assumption}\n"
            
            # 작업 분해
            tasks = work_plan.get("work_breakdown", [])
            result_text += f"• 총 작업 수: {len(tasks)}개\n\n"
            
            result_text += "📊 **작업별 상세 계획**\n\n"
            
            for i, task in enumerate(tasks, 1):
                task_id = task["id"]
                task_title = task["title"]
                task_agent = task["agent"]
                task_skills = task.get("skills", [])
                task_inputs = task.get("inputs", [])
                task_outputs = task.get("outputs", [])
                task_dod = task.get("dod", [])
                task_est_hours = task.get("est_hours", 0)
                task_dependencies = task.get("dependencies", [])
                task_parallelizable = task.get("parallelizable", False)
                
                result_text += f"**{i}. {task_title}** (ID: {task_id})\n"
                result_text += f"   • 담당 에이전트: {task_agent}\n"
                result_text += f"   • 필요 스킬: {', '.join(task_skills) if task_skills else 'N/A'}\n"
                result_text += f"   • 입력: {', '.join(task_inputs) if task_inputs else 'N/A'}\n"
                result_text += f"   • 출력: {', '.join(task_outputs) if task_outputs else 'N/A'}\n"
                result_text += f"   • 완료 기준: {', '.join(task_dod) if task_dod else 'N/A'}\n"
                result_text += f"   • 추정 시간: {task_est_hours}시간\n"
                result_text += f"   • 의존성: {', '.join(task_dependencies) if task_dependencies else '없음'}\n"
                result_text += f"   • 병렬 실행: {'가능' if task_parallelizable else '불가능'}\n"
                result_text += "\n"
            
            # 실행 계획
            execution_plan = work_plan.get("execution_plan", [])
            if execution_plan:
                result_text += "🚀 **실행 계획**\n"
                for i, phase in enumerate(execution_plan, 1):
                    phase_name = phase.get("phase", f"Phase {i}")
                    phase_targets = phase.get("targets", [])
                    phase_notes = phase.get("notes", "")
                    result_text += f"  {i}. **{phase_name}**: {', '.join(phase_targets)}\n"
                    if phase_notes:
                        result_text += f"     참고사항: {phase_notes}\n"
                result_text += "\n"
            
            # 리스크 및 완화 방안
            risks_global = work_plan.get("risks_global", [])
            mitigations_global = work_plan.get("mitigations_global", [])
            
            if risks_global or mitigations_global:
                result_text += "⚠️ **리스크 및 완화 방안**\n"
                if risks_global:
                    result_text += "• 리스크:\n"
                    for risk in risks_global:
                        result_text += f"  - {risk}\n"
                if mitigations_global:
                    result_text += "• 완화 방안:\n"
                    for mitigation in mitigations_global:
                        result_text += f"  - {mitigation}\n"
                result_text += "\n"
            
            # 사용자 요청 원문도 포함
            if hasattr(self, 'user_task') and self.user_task:
                result_text += f"📝 **원본 요청**: {self.user_task}\n"
            
            logger.info(f"format_planner_result completed, length: {len(result_text)}")
            return result_text
            
        except Exception as e:
            logger.error(f"Error in format_planner_result: {e}")
            return f"A2A 플래너 실행 완료!\n\n에이전트 수: {len(available_agents)}개\n작업 수: {len(work_plan.get('work_breakdown', []))}개\n\n상세 결과는 로그를 확인하세요."

    async def get_memory_data(self):
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(session_id=self.graph.session_id, order="Ascending", n_messages=self.n_messages)
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

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
                model_kwargs[input_.name] = getattr(self, f"{prefix}{input_.name}")
        return component.set(**model_kwargs).build_model()

    def set_component_params(self, component):
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            model_kwargs = {input_.name: getattr(self, f"{prefix}{input_.name}") for input_ in inputs}

            return component.set(**model_kwargs)
        return component

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            build_config.pop(field, None)

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
                # agent_llm의 input_types를 LanguageModel으로 강제 설정
                if key == "agent_llm":
                    build_config[key]["input_types"] = ["LanguageModel"]
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
                # agent_llm의 input_types를 LanguageModel으로 강제 설정
                if key == "agent_llm":
                    value.input_types = ["LanguageModel"]
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        # Iterate over all providers in the MODEL_PROVIDERS_DICT
        # Existing logic for updating build_config
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call the component class's update_build_config method
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

            provider_configs: dict[str, tuple[dict, list[dict]]] = {
                provider: (
                    MODEL_PROVIDERS_DICT[provider]["fields"],
                    [
                        MODEL_PROVIDERS_DICT[other_provider]["fields"]
                        for other_provider in MODEL_PROVIDERS_DICT
                        if other_provider != provider
                    ],
                )
                for provider in MODEL_PROVIDERS_DICT
            }
            if field_value in provider_configs:
                fields_to_add, fields_to_delete = provider_configs[field_value]

                # Delete fields from other providers
                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)

                # Add provider-specific fields
                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                # Reset input types for agent_llm
                build_config["agent_llm"]["input_types"] = ["LanguageModel"]
            elif field_value == "Custom":
                # Delete all provider fields
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
                # Update with custom component
                custom_component = DropdownInput(
                    name="agent_llm",
                    display_name="Language Model",
                    options=[*sorted(MODEL_PROVIDERS), "Custom"],
                    value="Custom",
                    real_time_refresh=True,
                    input_types=["LanguageModel"],
                    options_metadata=[MODELS_METADATA[key] for key in sorted(MODELS_METADATA.keys())]
                    + [{"icon": "brain"}],
                )
                build_config.update({"agent_llm": custom_component.to_dict()})
            # Update input types for all fields
            build_config = self.update_input_types(build_config)

            # Validate required keys
            default_keys = [
                "code",
                "_type",
                "agent_llm",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "handle_parsing_errors",
                "verbose",
            ]
            
            # 누락된 필수 키들을 기본값으로 설정
            if "tools" not in build_config:
                build_config["tools"] = {"value": "", "input_types": ["Tool"]}
            if "input_value" not in build_config:
                build_config["input_value"] = {"value": "", "input_types": ["Message"]}
            if "agent_description" not in build_config:
                build_config["agent_description"] = {"value": "A helpful assistant with access to the following tools:"}
            if "system_prompt" not in build_config:
                build_config["system_prompt"] = {"value": None, "input_types": ["Message"]}
            
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        if (
            isinstance(self.agent_llm, str)
            and self.agent_llm in MODEL_PROVIDERS_DICT
            and field_name in MODEL_DYNAMIC_UPDATE_FIELDS
        ):
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if provider_info:
                component_class = provider_info.get("component_class")
                component_class = self.set_component_params(component_class)
                prefix = provider_info.get("prefix")
                if component_class and hasattr(component_class, "update_build_config"):
                    # Call each component class's update_build_config method
                    # remove the prefix from the field_name
                    if isinstance(field_name, str) and isinstance(prefix, str):
                        field_name = field_name.replace(prefix, "")
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = _get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"
        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent", tool_description=description, callbacks=self.get_langchain_callbacks()
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)
        return tools
