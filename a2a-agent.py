import json
import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from uuid import uuid4
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
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_AI
from langflow.schema.message import Message


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



MODEL_PROVIDERS_LIST = ["Anthropic", "Google Generative AI", "Groq", "OpenAI"]


class A2AAgentComponent(ToolCallingAgentComponent):
    display_name: str = "A2A Agent"
    description: str = "A2A Discovery 서비스와 연동하여 멀티 에이전트 작업을 수행하는 에이전트"
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "A2A Agent"



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
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: A2A 에이전트의 초기 지시사항과 컨텍스트",
            value="""당신은 A2A Discovery 서비스와 연동하여 멀티 에이전트 작업을 수행하는 A2A 에이전트입니다.

주요 역할:
1. A2A Discovery 서비스에서 사용 가능한 에이전트 목록을 조회
2. 사용자 작업을 분석하고 작업 분할 계획 수립
3. 적절한 에이전트에게 작업 위임 및 실행
4. 결과 분석 및 개선안 제시 (Planning-Dispatcher-Critic 패턴)

작업 계획 수립 시 다음 원칙을 준수하세요:
- 목표를 명확한 산출물 중심으로 분해 (30-90분 내 완료 가능한 크기)
- 각 태스크에 담당 에이전트/필요 스킬/입출력/완료 기준/의존성/추정 시간 명시
- 위상정렬 가능한 순서의 실행 계획 제시
- 리스크와 대안 포함, 병렬화 가능 구간 명시
- OUTPUT_SCHEMA 형식으로만 응답""",
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
        # Tools 연결을 위한 입력 필드
        StrInput(
            name="tools",
            display_name="Tools",
            info="에이전트가 사용할 도구들",
            value="",   
            advanced=True,
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
        # 부모 클래스의 max_iterations를 무시하고 A2A 전용 사용
        IntInput(
            name="a2a_max_iterations",
            display_name="A2A Maximum Iterations",
            value=3,
            info="Planning-Dispatcher-Critic 사이클의 최대 반복 횟수 (기본값: 3회)",
            advanced=True,
        ),
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
        BoolInput(
            name="enable_event_bus",
            display_name="Enable Event Bus",
            advanced=True,
            info="이벤트 버스 기능을 활성화하여 태스크 간 상호작업을 관리",
            value=False,
        ),
        # Prompt Component 입력 필드들 (필수)
        HandleInput(
            name="work_plan_prompt",
            display_name="Work Plan Prompt (필수)",
            info="작업 계획 수립을 위한 프롬프트 (Prompt Template 컴포넌트 연결 필수)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),
        HandleInput(
            name="planning_prompt", 
            display_name="Planning Prompt (필수)",
            info="계획 검토 및 개선을 위한 프롬프트 (Prompt Template 컴포넌트 연결 필수)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),

        HandleInput(
            name="critic_prompt",
            display_name="Critic Prompt (필수)", 
            info="실행 결과 분석 및 비평을 위한 프롬프트 (Prompt Template 컴포넌트 연결 필수)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),

    ]
    outputs = [
        Output(name="response", display_name="Response", method="message_response")
    ]

    async def message_response(self) -> Message:
        """A2A Agent의 메인 응답 메서드"""
        logger.info("message_response called - starting A2A Agent execution")

        try:
            # LLM 초기화
            await self._initialize_llm()
            
            # 실제 A2A Agent 로직 실행
            result = await self.run_a2a_agent()
            return result
        except Exception as e:
            logger.error(f"A2A Agent execution error: {e}")
            return Message(
                text=f"A2A Agent 실행 중 오류가 발생했습니다: {str(e)}",
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
            
            if hasattr(self, 'llm') and self.llm is not None:
                logger.info("LLM already initialized")
                return
                
            # agent_llm이 설정되어 있는지 확인
            if hasattr(self, 'agent_llm') and self.agent_llm is not None:
                logger.info(f"Initializing LLM with agent_llm: {type(self.agent_llm)}")
                self.llm = self.agent_llm
            else:
                logger.warning("No agent_llm provided, LLM will not be available")
                self.llm = None
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    async def run_a2a_agent(self) -> Message:
        """A2A 에이전트의 메인 실행 로직"""
        try:
            # 1. A2A Discovery 서비스에서 사용 가능한 에이전트 목록 조회
            available_agents = await self.get_available_agents()
            logger.info(f"Available agents: {len(available_agents)} agents found")

            # 2. 작업 분석 및 계획 수립
            work_plan = await self.create_work_plan(available_agents)
            logger.info(f"Work plan created with {len(work_plan.get('work_breakdown', []))} tasks")

            # 3. Planning-Dispatcher-Critic 사이클 실행
            final_result = await self.execute_planning_dispatcher_critic_cycle(work_plan, available_agents)
            logger.info(f"Final result: {final_result}")

            # 4. 최종 결과 정리 및 반환
            user_friendly_result = self.format_final_result(work_plan, final_result)
            logger.info(f"Formatted result length: {len(user_friendly_result)}")
            logger.info(f"First 100 chars: {user_friendly_result[:100]}")
            
            # 결과가 비어있는 경우 백업 메시지 제공
            if not user_friendly_result or len(user_friendly_result.strip()) == 0:
                user_friendly_result = f"A2A 에이전트 실행 완료!\n\n작업 결과:\n{json.dumps(final_result, ensure_ascii=False, indent=2)}"
                logger.warning("Empty result detected, using backup message")
            
            return Message(
                text=user_friendly_result,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI
            )

        except Exception as e:
            logger.error(f"Error in A2A agent execution: {e}")
            return Message(
                text=f"A2A 에이전트 실행 중 오류가 발생했습니다: {str(e)}",
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI
            )

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

    def format_final_result(self, work_plan: Dict[str, Any], final_result: Dict[str, Any]) -> str:
        """최종 결과를 사용자 친화적인 형태로 포맷팅"""
        
        try:
            logger.info("Starting format_final_result")
            logger.info(f"work_plan keys: {list(work_plan.keys()) if work_plan else 'None'}")
            logger.info(f"final_result keys: {list(final_result.keys()) if final_result else 'None'}")
            
            # 헤더
            result_text = "🎯 **A2A 에이전트 실행 결과**\n\n"
            
            # 작업 개요
            tasks = work_plan.get("work_breakdown", [])
            execution_results = final_result.get("execution_results", {})
            iterations = final_result.get("iterations", 0)
            
            result_text += f"📋 **작업 개요**\n"
            result_text += f"• 총 작업 수: {len(tasks)}개\n"
            result_text += f"• 완료된 작업: {len(execution_results)}개\n"
            result_text += f"• 실행 반복: {iterations + 1}회\n\n"
            
            # 각 작업별 결과
            result_text += "📊 **작업별 실행 결과**\n\n"
            
            for i, task in enumerate(tasks, 1):
                task_id = task["id"]
                task_title = task["title"]
                task_agent = task["agent"]
                
                result_text += f"**{i}. {task_title}**\n"
                result_text += f"   • 담당: {task_agent}\n"
                
                if task_id in execution_results:
                    task_result = execution_results[task_id]
                    status = task_result.get("status", "unknown")
                    
                    if status == "completed":
                        result_text += f"   • 상태: ✅ 완료\n"
                        
                        # 결과 내용 추출
                        result_data = task_result.get("result", {})
                        
                        # 새로운 직접 실행 결과 처리
                        if isinstance(result_data, dict):
                            if "execution_method" in result_data:
                                execution_method = result_data["execution_method"]
                                result_content = result_data.get("result", "")
                                
                                if execution_method == "text_analysis":
                                    # 텍스트 분석 결과
                                    analysis = result_data.get("analysis", {})
                                    result_text += f"   • 실행 방법: 📊 텍스트 분석\n"
                                    result_text += f"   • 단어 수: {analysis.get('단어_수', 'N/A')}개\n"
                                    result_text += f"   • 문자 수: {analysis.get('문자_수_공백포함', 'N/A')}개\n"
                                    result_text += f"   • 문장 수: {analysis.get('문장_수', 'N/A')}개\n"
                                elif execution_method in ["llm_qa", "default_qa"]:
                                    # 질의응답 결과
                                    question = result_data.get("question", "")
                                    answer = result_data.get("answer", "")
                                    result_text += f"   • 실행 방법: ❓ 질의응답\n"
                                    result_text += f"   • 질문: {question[:50]}{'...' if len(question) > 50 else ''}\n"
                                    result_text += f"   • 답변: {answer[:100]}{'...' if len(answer) > 100 else ''}\n"
                                elif execution_method in ["llm_summary", "basic_summary"]:
                                    # 요약 결과
                                    original_length = len(result_data.get("original_text", ""))
                                    summary = result_data.get("summary", "")
                                    result_text += f"   • 실행 방법: 📝 요약\n"
                                    result_text += f"   • 원본 길이: {original_length}자\n"
                                    result_text += f"   • 요약: {summary[:100]}{'...' if len(summary) > 100 else ''}\n"
                                elif execution_method in ["llm_generation", "default_generation"]:
                                    # 생성 결과
                                    request = result_data.get("request", "")
                                    generated = result_data.get("generated_content", "")
                                    result_text += f"   • 실행 방법: ✨ 생성\n"
                                    result_text += f"   • 요청: {request[:50]}{'...' if len(request) > 50 else ''}\n"
                                    result_text += f"   • 생성 결과: {generated[:100]}{'...' if len(generated) > 100 else ''}\n"
                                elif execution_method in ["llm_general", "basic_general"]:
                                    # 일반 작업 결과
                                    task_desc = result_data.get("task_description", "")
                                    user_inp = result_data.get("user_input", "")
                                    result_text += f"   • 실행 방법: 🔧 일반 작업\n"
                                    result_text += f"   • 작업: {task_desc}\n"
                                    result_text += f"   • 입력: {user_inp[:50]}{'...' if len(user_inp) > 50 else ''}\n"
                                elif execution_method == "direct_calculation":
                                    # 직접 계산 결과
                                    calc_result = result_data.get("result", "")
                                    inputs = result_data.get("inputs", {})
                                    calculation = result_data.get("calculation", {})
                                    
                                    result_text += f"   • 실행 방법: 🧮 수학 계산\n"
                                    result_text += f"   • 결과: {calc_result}\n"
                                    if inputs:
                                        result_text += f"   • 입력값: 단어수 {inputs.get('word_count', 'N/A')}, 토큰수 {inputs.get('token_count', 'N/A')}\n"
                                else:
                                    # 기타 직접 실행 결과
                                    result_text += f"   • 실행 방법: {execution_method}\n"
                                    if isinstance(result_content, str):
                                        lines = result_content.strip().split('\n')
                                        result_text += f"   • 결과:\n"
                                        for line in lines[:3]:  # 첫 3줄만 표시
                                            result_text += f"     {line.strip()}\n"
                                        if len(lines) > 3:
                                            result_text += "     ...\n"
                                            
                            elif "agent_response" in result_data:
                                # A2A 에이전트 응답 (기존 로직)
                                agent_response = result_data["agent_response"]
                                if isinstance(agent_response, dict) and "result" in agent_response:
                                    response_content = agent_response["result"]
                                    if isinstance(response_content, dict) and "parts" in response_content:
                                        for part in response_content["parts"]:
                                            if part.get("kind") == "text":
                                                text_content = part.get("text", "")
                                                # 텍스트를 줄 단위로 나누어 처리
                                                lines = text_content.strip().split('\n')
                                                result_text += f"   • 실행 방법: 🤖 외부 에이전트\n"
                                                result_text += f"   • 결과:\n"
                                                for line in lines[:3]:  # 첫 3줄만 표시
                                                    result_text += f"     {line.strip()}\n"
                                                if len(lines) > 3:
                                                    result_text += "     ...\n"
                                                break
                        elif isinstance(result_data, str):
                            # 단순 문자열 결과
                            lines = result_data.strip().split('\n')
                            result_text += f"   • 결과:\n"
                            for line in lines[:3]:  # 첫 3줄만 표시
                                result_text += f"     {line.strip()}\n"
                            if len(lines) > 3:
                                result_text += "     ...\n"
                    elif status == "failed":
                        result_text += f"   • 상태: ❌ 실패\n"
                        error = task_result.get("error", "알 수 없는 오류")
                        result_text += f"   • 오류: {error}\n"
                    elif status == "skipped":
                        result_text += f"   • 상태: ⏭️ 건너뜀\n"
                        reason = task_result.get("reason", "")
                        result_text += f"   • 사유: {reason}\n"
                else:
                    result_text += f"   • 상태: ⏸️ 미실행\n"
                
                result_text += "\n"
            
            # 최종 요약
            completed_count = sum(1 for result in execution_results.values() if result.get("status") == "completed")
            failed_count = sum(1 for result in execution_results.values() if result.get("status") == "failed")
            
            result_text += "🏁 **최종 요약**\n"
            if completed_count == len(tasks) and failed_count == 0:
                result_text += "✅ 모든 작업이 성공적으로 완료되었습니다!\n"
            elif failed_count > 0:
                result_text += f"⚠️ {failed_count}개 작업이 실패했습니다. ({completed_count}/{len(tasks)} 완료)\n"
            else:
                result_text += f"🔄 작업이 진행 중입니다. ({completed_count}/{len(tasks)} 완료)\n"
            
            # 사용자 요청 원문도 포함
            if hasattr(self, 'user_task') and self.user_task:
                result_text += f"\n📝 **원본 요청**: {self.user_task}\n"
            
            logger.info(f"format_final_result completed, length: {len(result_text)}")
            return result_text
            
        except Exception as e:
            logger.error(f"Error in format_final_result: {e}")
            return f"A2A 에이전트 실행 완료!\n\n작업 완료: {len(final_result.get('execution_results', {}))}개\n상태: {final_result.get('status', 'unknown')}\n\n상세 결과는 로그를 확인하세요."

    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """A2A Discovery 서비스에서 사용 가능한 에이전트 목록을 조회"""
        try:
            api_url = f"{self.a2a_api_base}/agents?status=active"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        logger.info(f"A2A Discovery response: {response_data}")
                        
                        # A2A Discovery 서비스는 AgentListResponse 형태로 응답: {"agents": [...]}
                        if isinstance(response_data, dict) and 'agents' in response_data:
                            agents = response_data['agents']
                            logger.info(f"Successfully retrieved {len(agents)} agents from A2A Discovery service")
                            return agents
                        else:
                            logger.warning(f"Unexpected response format from A2A Discovery service: {type(response_data)}")
                            return []
                    else:
                        logger.warning(f"Failed to get agents from A2A Discovery service: {response.status}")
                        return []
        except Exception as e:
            logger.warning(f"Error connecting to A2A Discovery service: {e}")
            return []
        

    async def create_work_plan(self, available_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """사용자 작업을 분석하고 작업 분할 계획을 수립"""
        
        # Work Plan Prompt 필수 연결 확인
        if not hasattr(self, 'work_plan_prompt') or not self.work_plan_prompt:
            msg = "Work Plan Prompt가 연결되지 않았습니다. Prompt Template 컴포넌트를 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)
                
        planning_prompt = create_prompt_from_component(
            self.work_plan_prompt,
            default_system_message=self.system_prompt
        )
        
        if planning_prompt is None:
            msg = "Work Plan Prompt 변환에 실패했습니다. 올바른 프롬프트 템플릿을 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)

        # OUTPUT_SCHEMA 정의
        output_schema = {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "전체 목표"},
                "assumptions": {"type": "array", "items": {"type": "string"}, "description": "가정사항들"},
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
        
        # LLM이 있는지 확인하고 없으면 기본 계획 반환
        if not hasattr(self, 'llm') or self.llm is None:
            logger.warning("LLM not available, using default work plan")
            return self.get_default_work_plan(user_task, available_agents)

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
            
            logger.info("Work plan created successfully")
            
            # 수립된 작업 제목들 출력
            if "work_breakdown" in result:
                tasks = result["work_breakdown"]
                logger.info(f"수립된 작업 목록 ({len(tasks)}개):")
                for i, task in enumerate(tasks, 1):
                    title = task.get("title", f"Task {task.get('id', 'Unknown')}")
                    outputs = task.get("outputs", [])
                    dod = task.get("dod", [])
                    logger.info(f"  {i}. {title}\r\n     출력: {', '.join(outputs)}\r\n     완료 기준: {', '.join(dod)}")
            
                # 디버깅: 전체 작업 구조 로그
                logger.info(f"전체 작업 구조 디버깅:")
                for i, task in enumerate(tasks, 1):
                    logger.info(f"  Task {i} 상세: {task}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating work plan: {e}")
            raise

    async def execute_planning_dispatcher_critic_cycle(
        self, 
        work_plan: Dict[str, Any], 
        available_agents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Planning-Dispatcher-Critic 사이클을 실행하여 작업을 수행"""
        
        current_plan = work_plan
        iteration = 0
        execution_results = {}  # 초기화
        
        # A2A 전용 반복 횟수
        max_iter = self.a2a_max_iterations
        logger.info(f"Starting PDC cycle with A2A max_iterations: {max_iter}")
        
        while iteration < max_iter:
            logger.info(f"Starting iteration {iteration + 1} of {max_iter}")
            
            try:
                # 1. Planning: 현재 계획 검토 및 개선
                proposed_improved_plan = await self.planning_phase(current_plan, available_agents)
                
                # Planning Phase 결과 검증
                original_task_count = len(current_plan.get("work_breakdown", []))
                improved_task_count = len(proposed_improved_plan.get("work_breakdown", []))
                
                if improved_task_count != original_task_count:
                    logger.warning(f"Planning Phase가 작업 개수를 변경했습니다: {original_task_count} → {improved_task_count}")
                    logger.warning("원본 계획을 유지합니다.")
                    improved_plan = current_plan
                else:
                    improved_plan = proposed_improved_plan
                
                # 2. Dispatcher: 작업을 적절한 에이전트에게 위임
                logger.info("Starting dispatcher phase...")
                new_results = await self.dispatcher_phase(improved_plan, available_agents, execution_results)
                execution_results.update(new_results)
                logger.info(f"Dispatcher phase returned. Total execution results: {len(execution_results)}")
                
                # 3. Critic: 결과 분석 및 개선안 제시
                logger.info("Starting critic phase...")
                critique_result = await self.critic_phase(execution_results, improved_plan)
                logger.info(f"Critic phase completed. Critique result: {critique_result}")

                # 4. 다음 반복을 위한 계획 업데이트
                proposed_plan = critique_result.get("updated_plan", improved_plan)
                
                # 작업 ID 일관성 검증
                current_task_ids = {task["id"] for task in improved_plan.get("work_breakdown", [])}
                proposed_task_ids = {task["id"] for task in proposed_plan.get("work_breakdown", [])}
                
                # 이미 처리된 작업의 ID가 변경되었는지 확인
                processed_task_ids = set(execution_results.keys())
                id_conflicts = processed_task_ids - proposed_task_ids
                
                if id_conflicts:
                    logger.warning(f"Critic이 이미 처리된 작업 ID를 변경했습니다: {id_conflicts}")
                    logger.warning("기존 계획을 유지합니다.")
                    current_plan = improved_plan
                else:
                    current_plan = proposed_plan
                    
                logger.info(f"Updated plan: {current_plan}")

                # 5. 완료 조건 확인
                if critique_result.get("is_complete", False):
                    logger.info(f"Task completed after {iteration + 1} iterations")
                    break
                    
                iteration += 1
                
                # 6. 잠시 대기 (과도한 API 호출 방지)
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {e}")
                break
        
        logger.info(f"PDC cycle completed after {iteration} iterations (max: {max_iter})")
        
        return {
            # "final_plan": current_plan,
            "execution_results": execution_results,
            "iterations": iteration,
            "status": "completed" if iteration < max_iter else "max_iterations_reached"
        }

    async def planning_phase(self, current_plan: Dict[str, Any], available_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """계획 단계: 현재 계획을 검토하고 개선"""
        
        # Planning Prompt 필수 연결 확인
        if not hasattr(self, 'planning_prompt') or not self.planning_prompt:
            msg = "Planning Prompt가 연결되지 않았습니다. Prompt Template 컴포넌트를 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info("Using external planning prompt from Prompt Component")
        planning_prompt = create_prompt_from_component(
            self.planning_prompt,
            default_system_message="당신은 작업 계획을 검토하고 개선하는 계획 전문가입니다."
        )
        
        if planning_prompt is None:
            msg = "Planning Prompt 변환에 실패했습니다. 올바른 프롬프트 템플릿을 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)
        
        # LLM이 있는지 확인하고 없으면 현재 계획 반환
        if not hasattr(self, 'llm') or self.llm is None:
            logger.warning("LLM not available for planning phase, using current plan")
            return current_plan

        try:
            chain = planning_prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({
                "current_plan": json.dumps(current_plan, ensure_ascii=False, indent=2),
                "available_agents": json.dumps(available_agents, ensure_ascii=False, indent=2)
            })
            return result
        except Exception as e:
            logger.warning(f"Planning phase failed: {e}, using current plan")
            return current_plan

    async def dispatcher_phase(self, plan: Dict[str, Any], available_agents: List[Dict[str, Any]], existing_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """디스패처 단계: 작업을 적절한 에이전트에게 위임"""
        
        # 기존 결과가 있으면 사용, 없으면 빈 딕셔너리로 시작
        if existing_results is None:
            existing_results = {}
        
        new_execution_results = {}
        
        # 디버깅: 현재 상태 로그
        all_tasks = plan.get("work_breakdown", [])
        logger.info(f"Dispatcher: 총 {len(all_tasks)}개 작업, 기존 결과: {list(existing_results.keys()) if existing_results else '없음'}")
        
        # 아직 처리되지 않은 첫 번째 작업만 처리
        for task in all_tasks:
            task_id = task["id"]
            agent_id = task["agent"]
            
            logger.info(f"Dispatcher: 작업 {task_id} 확인 중... (에이전트: {agent_id})")
            
            # 이미 처리된 작업은 건너뛰기
            if task_id in existing_results:
                logger.info(f"Task {task_id} already processed, skipping")
                continue
                
            logger.info(f"Processing task {task_id} with agent {agent_id}")
            
            # 이전 작업 결과에서 현재 작업의 필요 출력이 이미 있는지 확인
            required_outputs = task.get("outputs", [])
            if required_outputs and existing_results:
                available_outputs = self.extract_available_outputs(existing_results)
                if all(output in available_outputs for output in required_outputs):
                    logger.info(f"Task {task_id} 스킵: 필요한 출력 {required_outputs}이 이미 이전 작업에서 생성됨")
                    # 이전 결과에서 해당 출력을 가져와서 결과로 설정
                    combined_result = self.combine_previous_outputs(existing_results, required_outputs)
                    new_execution_results[task_id] = {
                        "status": "skipped",
                        "result": combined_result,
                        "agent": "previous_results",
                        "reason": f"Required outputs {required_outputs} already available from previous tasks"
                    }
                    logger.info(f"Task {task_id} added as skipped: {new_execution_results[task_id]}")
                    break
            
            try:
                # 에이전트에게 작업 위임
                if agent_id != "self":
                    logger.info(f"Delegating task {task_id} to agent {agent_id}")
                    result = await self.delegate_task_to_agent(task, agent_id, available_agents)
                    logger.info(f"Task {task_id} delegation result: {result}")
                else:
                    # 직접 실행
                    logger.info(f"Executing task {task_id} directly")
                    result = await self.execute_task_directly(task, existing_results)
                    logger.info(f"Task {task_id} direct execution result: {result}")
                
                new_execution_results[task_id] = {
                    "status": "completed",
                    "result": result,
                    "agent": agent_id
                }
                
                logger.info(f"Task {task_id} added to new_execution_results: {new_execution_results[task_id]}")
                
                # 한 번에 하나의 작업만 처리하고 반환 (PDC 사이클을 위해)
                break
                
            except Exception as e:
                logger.error(f"Task {task_id} execution failed: {e}")
                new_execution_results[task_id] = {
                    "status": "failed",
                    "error": str(e),
                    "agent": agent_id
                }
                # 실패한 경우에도 한 번에 하나만 처리
                break
        
        # 모든 작업을 확인했지만 처리할 것이 없는 경우
        if len(new_execution_results) == 0:
            logger.info(f"Dispatcher: 처리할 새로운 작업이 없습니다. (총 {len(all_tasks)}개 작업 중 {len(existing_results)}개 완료)")
        
        logger.info(f"Dispatcher phase completed. Returning {len(new_execution_results)} new results")
        return new_execution_results

    def extract_available_outputs(self, execution_results: Dict[str, Any]) -> List[str]:
        """이전 작업 결과에서 사용 가능한 출력 목록을 추출"""
        available_outputs = []
        
        for task_id, result in execution_results.items():
            if result.get("status") == "completed" and "agent_response" in result:
                agent_response = result["agent_response"]
                if "result" in agent_response and "parts" in agent_response["result"]:
                    # 응답 텍스트에서 출력 유형을 추정
                    for part in agent_response["result"]["parts"]:
                        if "text" in part:
                            text = part["text"].lower()
                            # 일반적인 출력 패턴 감지
                            if "단어" in text and ("수" in text or "개수" in text):
                                available_outputs.append("단어 수")
                                available_outputs.append("단어 목록")
                            if "토큰" in text and ("수" in text or "개수" in text):
                                available_outputs.append("토큰 수")
                                available_outputs.append("토큰 목록")
                            if "성격" in text or "감정" in text:
                                available_outputs.append("문장 성격")
        
        logger.info(f"사용 가능한 출력: {available_outputs}")
        return available_outputs

    def combine_previous_outputs(self, execution_results: Dict[str, Any], required_outputs: List[str]) -> Dict[str, Any]:
        """이전 작업 결과에서 필요한 출력들을 조합하여 반환"""
        combined_result = {
            "task_id": "combined_from_previous",
            "agent_id": "result_combiner",
            "status": "completed",
            "timestamp": "2024-01-01T00:00:00Z",
            "agent_response": {
                "jsonrpc": "2.0",
                "id": "combined",
                "result": {
                    "kind": "message",
                    "messageId": str(uuid4()),
                    "role": "agent",
                    "parts": [{
                        "kind": "text",
                        "text": f"이전 작업 결과에서 필요한 출력 {required_outputs}을 조합했습니다. 원본 결과를 참조하세요."
                    }]
                }
            }
        }
        
        return combined_result

    async def execute_task_directly(self, task: Dict[str, Any], existing_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """직접 실행: 수학 계산 등 내장 기능으로 처리 가능한 작업"""
        
        task_id = task["id"]
        task_title = task["title"]
        task_inputs = task.get("inputs", [])
        task_outputs = task.get("outputs", [])
        
        logger.info(f"직접 실행 작업: {task_title}")
        logger.info(f"필요한 입력: {task_inputs}")
        logger.info(f"기대 출력: {task_outputs}")
        
        # 수학 계산 작업 처리
        if "수학 계산" in task_title or "곱셈" in task_title or "자연로그" in task_title:
            if existing_results:
                # 이전 작업 결과에서 필요한 데이터 추출
                word_count = None
                token_count = None
                
                for prev_task_id, prev_result in existing_results.items():
                    if prev_result.get("status") == "completed":
                        result_data = prev_result.get("result", {})
                        if isinstance(result_data, dict) and "agent_response" in result_data:
                            # A2A 에이전트 응답에서 데이터 추출
                            agent_response = result_data["agent_response"]
                            if isinstance(agent_response, dict) and "result" in agent_response:
                                response_text = agent_response["result"]
                                if isinstance(response_text, dict) and "parts" in response_text:
                                    for part in response_text["parts"]:
                                        if part.get("kind") == "text":
                                            text_content = part.get("text", "")
                                            # 텍스트에서 단어 수와 토큰 수 추출
                                            import re
                                            word_match = re.search(r'단어.*?(\d+)', text_content)
                                            token_match = re.search(r'토큰.*?(\d+)', text_content)
                                            
                                            if word_match:
                                                word_count = int(word_match.group(1))
                                            if token_match:
                                                token_count = int(token_match.group(1))
                
                if word_count is not None and token_count is not None:
                    import math
                    product = word_count * token_count
                    natural_log = math.log(product)
                    
                    result = {
                        "task_id": task_id,
                        "execution_method": "direct_calculation",
                        "inputs": {
                            "word_count": word_count,
                            "token_count": token_count
                        },
                        "calculation": {
                            "product": product,
                            "natural_log": natural_log
                        },
                        "result": f"단어 수({word_count}) × 토큰 수({token_count}) = {product}, ln({product}) = {natural_log:.6f}"
                    }
                    logger.info(f"수학 계산 완료: {result['result']}")
                    return result
                else:
                    logger.warning(f"이전 작업 결과에서 단어 수({word_count}) 또는 토큰 수({token_count})를 찾을 수 없습니다.")
            else:
                logger.warning("수학 계산을 위한 이전 작업 결과가 없습니다.")
        
        # 일반적인 작업 처리
        try:
            # 사용자 입력 처리
            user_input = self.input_value.content if hasattr(self.input_value, 'content') else str(self.input_value)
            
            # 작업 유형에 따른 처리
            if "분석" in task_title or "텍스트" in task_title:
                return await self.handle_text_analysis_task(task, user_input)
            elif "질문" in task_title or "답변" in task_title or "QA" in task_title:
                return await self.handle_qa_task(task, user_input)
            elif "요약" in task_title or "정리" in task_title:
                return await self.handle_summary_task(task, user_input)
            elif "생성" in task_title or "작성" in task_title:
                return await self.handle_generation_task(task, user_input)
            else:
                # 기본 처리: LLM을 사용한 일반적인 대응
                return await self.handle_general_task(task, user_input)
                
        except Exception as e:
            logger.error(f"작업 실행 중 오류 발생: {e}")
            return {
                "task_id": task_id,
                "execution_method": "direct",
                "status": "error",
                "error": str(e),
                "result": f"작업 '{task_title}' 실행 중 오류가 발생했습니다: {str(e)}"
            }

    async def handle_text_analysis_task(self, task: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """텍스트 분석 작업 처리"""
        logger.info("텍스트 분석 작업 실행")
        
        # 기본 텍스트 분석 수행
        word_count = len(user_input.split())
        char_count = len(user_input)
        char_count_no_spaces = len(user_input.replace(' ', ''))
        
        # 문장 수 계산
        import re
        sentences = re.split(r'[.!?]+', user_input)
        sentence_count = len([s for s in sentences if s.strip()])
        
        # 단락 수 계산
        paragraphs = user_input.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])
        
        analysis_result = {
            "텍스트": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "단어_수": word_count,
            "문자_수_공백포함": char_count,
            "문자_수_공백제외": char_count_no_spaces,
            "문장_수": sentence_count,
            "단락_수": paragraph_count
        }
        
        result_text = f"""📊 텍스트 분석 결과:
        
🔤 단어 수: {word_count}개
📝 문자 수 (공백 포함): {char_count}개
✏️ 문자 수 (공백 제외): {char_count_no_spaces}개
📄 문장 수: {sentence_count}개
📑 단락 수: {paragraph_count}개

📖 분석된 텍스트: "{user_input[:100]}{'...' if len(user_input) > 100 else ''}"
        """
        
        return {
            "task_id": task["id"],
            "execution_method": "text_analysis",
            "status": "completed",
            "analysis": analysis_result,
            "result": result_text
        }

    async def handle_qa_task(self, task: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """질의응답 작업 처리"""
        logger.info("질의응답 작업 실행")
        
        # LLM이 있는 경우 LLM을 사용
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage
                
                prompt = f"""다음 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요:

질문: {user_input}

답변은 명확하고 이해하기 쉽게 작성해주세요."""
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                answer = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "task_id": task["id"],
                    "execution_method": "llm_qa",
                    "status": "completed",
                    "question": user_input,
                    "answer": answer,
                    "result": f"❓ 질문: {user_input}\n\n💡 답변: {answer}"
                }
            except Exception as e:
                logger.error(f"LLM 질의응답 실행 중 오류: {e}")
        
        # LLM이 없거나 오류가 발생한 경우 기본 응답
        default_answer = f"'{user_input}'에 대한 질문을 받았습니다. 더 구체적인 답변을 위해서는 LLM 모델이 필요합니다."
        
        return {
            "task_id": task["id"],
            "execution_method": "default_qa",
            "status": "completed",
            "question": user_input,
            "answer": default_answer,
            "result": f"❓ 질문: {user_input}\n\n💡 답변: {default_answer}"
        }

    async def handle_summary_task(self, task: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """요약 작업 처리"""
        logger.info("요약 작업 실행")
        
        # LLM이 있는 경우 LLM을 사용
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage
                
                prompt = f"""다음 텍스트를 핵심 내용을 포함하여 간결하게 요약해주세요:

원본 텍스트:
{user_input}

요약은 3-5개의 주요 포인트로 작성해주세요."""
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                summary = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "task_id": task["id"],
                    "execution_method": "llm_summary",
                    "status": "completed",
                    "original_text": user_input,
                    "summary": summary,
                    "result": f"📄 원본 길이: {len(user_input)}자\n\n📝 요약:\n{summary}"
                }
            except Exception as e:
                logger.error(f"LLM 요약 실행 중 오류: {e}")
        
        # LLM이 없거나 오류가 발생한 경우 기본 요약
        words = user_input.split()
        if len(words) > 50:
            # 첫 30단어와 마지막 20단어로 기본 요약
            summary = " ".join(words[:30]) + " ... " + " ".join(words[-20:])
        else:
            summary = user_input
        
        return {
            "task_id": task["id"],
            "execution_method": "basic_summary",
            "status": "completed",
            "original_text": user_input,
            "summary": summary,
            "result": f"📄 원본 길이: {len(user_input)}자\n\n📝 기본 요약:\n{summary}"
        }

    async def handle_generation_task(self, task: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """생성 작업 처리"""
        logger.info("생성 작업 실행")
        
        # LLM이 있는 경우 LLM을 사용
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage
                
                prompt = f"""다음 요청에 따라 창의적이고 유용한 내용을 생성해주세요:

요청: {user_input}

적절한 형식과 구조로 작성해주세요."""
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                generated_content = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "task_id": task["id"],
                    "execution_method": "llm_generation",
                    "status": "completed",
                    "request": user_input,
                    "generated_content": generated_content,
                    "result": f"🎯 요청: {user_input}\n\n✨ 생성된 내용:\n{generated_content}"
                }
            except Exception as e:
                logger.error(f"LLM 생성 실행 중 오류: {e}")
        
        # LLM이 없거나 오류가 발생한 경우 기본 응답
        default_content = f"'{user_input}' 요청을 받았습니다. 더 정교한 생성을 위해서는 LLM 모델이 필요합니다."
        
        return {
            "task_id": task["id"],
            "execution_method": "default_generation",
            "status": "completed",
            "request": user_input,
            "generated_content": default_content,
            "result": f"🎯 요청: {user_input}\n\n✨ 기본 응답:\n{default_content}"
        }

    async def handle_general_task(self, task: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        """일반적인 작업 처리"""
        logger.info("일반 작업 실행")
        
        # 기본적인 작업 처리
        task_description = task.get("title", "일반 작업")
        
        # LLM이 있는 경우 LLM을 사용
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                from langchain_core.messages import HumanMessage
                
                prompt = f"""다음 작업을 수행해주세요:

작업: {task_description}
사용자 입력: {user_input}

적절한 방식으로 작업을 완료하고 결과를 제공해주세요."""
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                result_content = response.content if hasattr(response, 'content') else str(response)
                
                return {
                    "task_id": task["id"],
                    "execution_method": "llm_general",
                    "status": "completed",
                    "task_description": task_description,
                    "user_input": user_input,
                    "result": f"🔧 작업: {task_description}\n📝 입력: {user_input}\n\n✅ 결과:\n{result_content}"
                }
            except Exception as e:
                logger.error(f"LLM 일반 작업 실행 중 오류: {e}")
        
        # LLM이 없거나 오류가 발생한 경우 기본 처리
        result_content = f"'{task_description}' 작업을 '{user_input}' 입력으로 처리했습니다. 더 정교한 처리를 위해서는 LLM 모델이나 전용 도구가 필요합니다."
        
        return {
            "task_id": task["id"],
            "execution_method": "basic_general",
            "status": "completed",
            "task_description": task_description,
            "user_input": user_input,
            "result": f"🔧 작업: {task_description}\n📝 입력: {user_input}\n\n✅ 기본 처리 결과:\n{result_content}"
        }

    async def delegate_task_to_agent(self, task: Dict[str, Any], agent_id: str, available_agents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """특정 에이전트에게 작업을 위임"""
        
        logger.info(f"Delegating task {task['id']} to agent {agent_id}")
        logger.info(f">> Task : {task['title']}")
        logger.info(f">> Task Debug: {task}")  # 디버깅: 전체 task 정보 출력
        
        logger.info(f">> Available Agents : {len(available_agents)}")
        
        # 계획에서 이미 할당된 에이전트를 직접 매칭 (에이전트 평가 단계는 Work Plan에서 완료됨)
        logger.info(f"Finding pre-assigned agent: {agent_id}")
        target_agent = None
        
        for agent in available_agents:
            if agent.get("id") == agent_id:
                target_agent = agent
                logger.info(f"Found target agent: {agent.get('name', agent_id)}")
                break
        
        if not target_agent:
            logger.error(f"Pre-assigned agent '{agent_id}' not found in available agents")
            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "status": "failed",
                "error": f"Pre-assigned agent '{agent_id}' not found in available agents",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        
        # 에이전트 URL 확인 및 변환
        agent_url = target_agent.get("url") or target_agent.get("endpoint")
        if not agent_url:
            agent_name = target_agent.get("name", target_agent.get("id", "Unknown"))
            msg = f"선택된 에이전트 '{agent_name}' (ID: {target_agent.get('id', 'Unknown')})에 URL이 지정되지 않았습니다. 에이전트 설정을 확인해주세요."
            logger.error(msg)
            raise ValueError(msg)
        
        # 0.0.0.0을 127.0.0.1로 변환 (클라이언트 호출용)
        if "0.0.0.0" in agent_url:
            agent_url = agent_url.replace("0.0.0.0", "127.0.0.1")
            logger.info(f"Converted 0.0.0.0 to 127.0.0.1: {agent_url}")
        
        # A2A 프로토콜에 맞는 실제 API 호출
        try:
            logger.info(f"Calling A2A API: {agent_url}")
            
            # A2A JSON-RPC 2.0 프로토콜 페이로드 구성
            a2a_rpc_payload = {
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
                                "text": f"Task: {task['title']}\nDescription: {task.get('description', '')}\nInputs: {', '.join(task.get('inputs', []))}\nExpected Outputs: {', '.join(task.get('outputs', []))}"
                            }
                        ]
                    },
                    "stream": False  # 스트리밍 비활성화
                }
            }
            
            # HTTP 헤더 설정
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "A2A-Agent-Component/1.0"
            }
            
            logger.info(f"A2A RPC payload: {a2a_rpc_payload}")
            
            # A2A 표준 API 호출
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    agent_url,
                    json=a2a_rpc_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    logger.info(f"A2A API response status: {response.status}")
                    
                    if response.status == 200:
                        try:
                            result_data = await response.json()
                            logger.info(f"Task {task['id']} successfully delegated to agent {agent_id}")
                            
                            # JSON-RPC 2.0 응답 구조 처리
                            if "result" in result_data:
                                rpc_result = result_data["result"]
                                # 응답 수신 확인 로그
                                logger.info(f"Agent response received from {agent_id}")
                                
                                final_result = {
                                    "task_id": task["id"],
                                    "agent_id": agent_id,
                                    "status": "completed",
                                    "timestamp": "2024-01-01T00:00:00Z",
                                    "agent_response": result_data
                                }
                                return final_result
                            elif "error" in result_data:
                                logger.error(f"Found 'error' in response: {result_data['error']}")
                                return {
                                    "task_id": task["id"],
                                    "agent_id": agent_id,
                                    "status": "failed",
                                    "error": f"RPC Error: {result_data['error']}",
                                    "timestamp": "2024-01-01T00:00:00Z"
                                }
                            else:
                                logger.warning(f"No 'result' or 'error' found in response, using raw data")
                                return {
                                    "task_id": task["id"],
                                    "agent_id": agent_id,
                                    "status": "completed",
                                    "result": result_data,
                                    "timestamp": "2024-01-01T00:00:00Z",
                                    "agent_response": result_data
                                }
                        except Exception as json_error:
                            logger.error(f"Error parsing JSON response: {json_error}")
                            response_text = await response.text()
                            logger.error(f"Raw response text: {response_text}")
                            return {
                                "task_id": task["id"],
                                "agent_id": agent_id,
                                "status": "failed",
                                "error": f"JSON parsing error: {json_error}",
                                "timestamp": "2024-01-01T00:00:00Z"
                            }
                    else:
                        error_text = await response.text()
                        logger.error(f"Agent {agent_id} returned error: {response.status} - {error_text}")
                        
                        return {
                            "task_id": task["id"],
                            "agent_id": agent_id,
                            "status": "failed",
                            "error": f"Agent returned HTTP {response.status}: {error_text}",
                            "timestamp": "2024-01-01T00:00:00Z"
                        }
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout while delegating task {task['id']} to agent {agent_id}")
            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "status": "timeout",
                "error": "Request timeout",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        except Exception as e:
            logger.error(f"Error delegating task {task['id']} to agent {agent_id}: {e}")
            return {
                "task_id": task["id"],
                "agent_id": agent_id,
                "status": "error",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z"
            }



    async def critic_phase(self, execution_results: Dict[str, Any], current_plan: Dict[str, Any]) -> Dict[str, Any]:
        """비평 단계: 실행 결과를 분석하고 개선안 제시"""
        
        logger.info("=== CRITIC PHASE STARTED ===")
        logger.info(f"Analyzing {len(execution_results)} execution results")
        
        # Critic Prompt 필수 연결 확인
        if not hasattr(self, 'critic_prompt') or not self.critic_prompt:
            msg = "Critic Prompt가 연결되지 않았습니다. Prompt Template 컴포넌트를 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info("Using external critic prompt from Prompt Component for LLM-based analysis")
        return await self._llm_based_critic(execution_results, current_plan)
    
    async def _llm_based_critic(self, execution_results: Dict[str, Any], current_plan: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 기반 비평 분석"""
        
        # 외부 Prompt Component 사용
        critic_prompt = create_prompt_from_component(
            self.critic_prompt,
            default_system_message="당신은 작업 실행 결과를 분석하고 개선안을 제시하는 전문 비평가입니다."
        )
        
        if critic_prompt is None:
            msg = "Critic Prompt 변환에 실패했습니다. 올바른 프롬프트 템플릿을 연결해주세요."
            logger.error(msg)
            raise ValueError(msg)
        
        try:
            # 실행 결과 요약 생성
            total_tasks = len(current_plan.get("work_breakdown", []))
            completed_tasks = sum(1 for result in execution_results.values() if result.get("status") == "completed")
            failed_tasks = sum(1 for result in execution_results.values() if result.get("status") == "failed")
            
            # 현재 진행 상황 컨텍스트 추가
            all_task_ids = [task["id"] for task in current_plan.get("work_breakdown", [])]
            completed_task_ids = [task_id for task_id, result in execution_results.items() if result.get("status") == "completed"]
            pending_task_ids = [task_id for task_id in all_task_ids if task_id not in execution_results]
            
            task_summary = {
                "total": total_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "completion_rate": f"{completed_tasks}/{total_tasks}",
                "execution_context": {
                    "mode": "순차 실행 (PDC 사이클)",
                    "current_status": f"현재까지 {completed_tasks}개 작업 완료, {len(pending_task_ids)}개 작업 대기 중",
                    "completed_tasks": completed_task_ids,
                    "pending_tasks": pending_task_ids,
                    "note": "작업은 한 번에 하나씩 순차적으로 처리되며, 각 작업 완료 후 결과를 분석합니다."
                }
            }
            
            # 실행 결과 요약 - 큰 응답 데이터 처리
            logger.info(f"Processing {len(execution_results)} execution results for critic analysis")
            summarized_results = {}
            for task_id, result in execution_results.items():
                logger.info(f"Processing result for task {task_id}: keys={list(result.keys())}")
                
                # execution_results의 중첩 구조 처리
                # result = {'status': 'completed', 'result': {...actual_data...}, 'agent': '...'}
                actual_result = result.get("result", result)  # 중첩된 result 구조 해결
                
                summary = {
                    "task_id": actual_result.get("task_id", task_id),
                    "agent_id": actual_result.get("agent_id", result.get("agent", "unknown")),
                    "status": actual_result.get("status", result.get("status", "unknown")),
                    "timestamp": actual_result.get("timestamp", "unknown")
                }
                
                # result 데이터 처리 (A2A 에이전트 응답 또는 직접 계산 결과)
                has_agent_response = "agent_response" in actual_result
                has_result_in_response = has_agent_response and actual_result["agent_response"] and "result" in actual_result["agent_response"]
                has_direct_result = "result" in actual_result and "execution_method" in actual_result
                
                logger.info(f"Task {task_id} - has_agent_response: {has_agent_response}, has_result_in_response: {has_result_in_response}")
                logger.info(f"Task {task_id} - has_direct_result: {has_direct_result}")
                
                if has_result_in_response:
                    # A2A 에이전트 응답 처리
                    agent_result = actual_result["agent_response"]["result"]
                    result_str = str(agent_result)
                    summary["result_preview"] = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    summary["result_size"] = len(result_str)
                    logger.info(f"Task {task_id} - result_size: {len(result_str)}")
                elif has_direct_result:
                    # 직접 실행 결과 처리 (수학 계산 등)
                    direct_result = actual_result["result"]
                    result_str = str(direct_result)
                    summary["result_preview"] = result_str[:200] + "..." if len(result_str) > 200 else result_str
                    summary["result_size"] = len(result_str)
                    summary["execution_method"] = actual_result.get("execution_method", "direct")
                    logger.info(f"Task {task_id} - direct result_size: {len(result_str)}")
                else:
                    summary["result_preview"] = "No result data"
                    logger.warning(f"Task {task_id} - No result data found")
                
                if "error" in result:
                    summary["error"] = result["error"]
                    
                summarized_results[task_id] = summary
            
            # LLM이 있는지 확인하고 없으면 휴리스틱 기반 비평 사용
            if not hasattr(self, 'llm') or self.llm is None:
                logger.warning("LLM not available for critic phase, using heuristic critic")
                return await self._heuristic_based_critic(execution_results, current_plan)

            chain = critic_prompt | self.llm | JsonOutputParser()
            result = await chain.ainvoke({
                "execution_results": json.dumps(summarized_results, ensure_ascii=False, indent=2),
                "current_plan": json.dumps(current_plan, ensure_ascii=False, indent=2),
                "task_summary": json.dumps(task_summary, ensure_ascii=False, indent=2)
            })
            
            # 기본값 보장
            if "is_complete" not in result:
                result["is_complete"] = (completed_tasks == total_tasks) and (failed_tasks == 0)
            if "updated_plan" not in result:
                result["updated_plan"] = current_plan
            if "next_actions" not in result:
                result["next_actions"] = ["계속 진행"]
                
            logger.info("LLM-based critic analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"LLM critic analysis failed: {e}, falling back to heuristic")
            return await self._heuristic_based_critic(execution_results, current_plan)
    
    async def _heuristic_based_critic(self, execution_results: Dict[str, Any], current_plan: Dict[str, Any]) -> Dict[str, Any]:
        """휴리스틱 기반 비평 분석 (기존 로직)"""
        
        # 실행 결과 분석
        total_tasks = len(current_plan.get("work_breakdown", []))
        completed_tasks = 0
        failed_tasks = 0
        
        for task_id, result in execution_results.items():
            if result.get("status") == "completed":
                completed_tasks += 1
            elif result.get("status") == "failed":
                failed_tasks += 1
        
        logger.info(f"Task analysis: {completed_tasks}/{total_tasks} completed, {failed_tasks} failed")
        
        # 완료 조건 판단
        is_complete = (completed_tasks == total_tasks) and (failed_tasks == 0)
        
        # 간단한 비평 생성
        if is_complete:
            critique = f"모든 작업이 성공적으로 완료되었습니다. ({completed_tasks}/{total_tasks} 완료)"
            next_actions = ["작업 완료"]
        elif failed_tasks > 0:
            critique = f"일부 작업이 실패했습니다. ({failed_tasks}개 실패, {completed_tasks}개 완료)"
            next_actions = ["실패한 작업 재시도", "오류 원인 분석"]
        else:
            critique = f"작업이 진행 중입니다. ({completed_tasks}/{total_tasks} 완료)"
            next_actions = ["남은 작업 계속 진행"]
        
        result = {
            "critique": critique,
            "updated_plan": current_plan,  # 현재 계획 유지
            "is_complete": is_complete,
            "next_actions": next_actions,
            "task_summary": {
                "total": total_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks
            }
        }
        
        logger.info(f"Heuristic critic phase result: {result}")
        return result

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
                "a2a_max_iterations",
                "handle_parsing_errors",
                "verbose",
                "work_plan_prompt",
                "planning_prompt", 

                "critic_prompt",
            ]
            
            # 누락된 필수 키들을 기본값으로 설정
            if "tools" not in build_config:
                build_config["tools"] = {"value": "", "input_types": ["Tool"]}
            if "input_value" not in build_config:
                build_config["input_value"] = {"value": "", "input_types": ["Message"]}
            if "agent_description" not in build_config:
                build_config["agent_description"] = {"value": "A helpful assistant with access to the following tools:"}
            if "work_plan_prompt" not in build_config:
                build_config["work_plan_prompt"] = {"value": None, "input_types": ["Message"]}
            if "planning_prompt" not in build_config:
                build_config["planning_prompt"] = {"value": None, "input_types": ["Message"]}

            if "critic_prompt" not in build_config:
                build_config["critic_prompt"] = {"value": None, "input_types": ["Message"]}
            
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



