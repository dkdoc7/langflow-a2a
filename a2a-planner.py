import json
import asyncio
from typing import List, Dict, Any
import aiohttp
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser

from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_PROVIDERS,
    MODEL_PROVIDERS_DICT,
    MODELS_METADATA,
)
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.utils import update_component_build_config
from langflow.io import BoolInput, DropdownInput, HandleInput, IntInput, MultilineInput, Output, StrInput
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_AI


def create_prompt_from_component(prompt_component, default_system_message: str = "") -> ChatPromptTemplate:
    """
    프롬프트 컴포넌트에서 ChatPromptTemplate을 생성하는 헬퍼 함수
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
        
        # ChatPromptTemplate 생성
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


class A2APlannerComponent(ToolCallingAgentComponent):
    display_name: str = "A2A Planner"
    description: str = "A2A Discovery 서비스와 연동하여 에이전트 목록을 조회하고 작업 계획을 수립하는 플래너"
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "A2A Planner"

    inputs = [
        # LLM 연결을 위한 입력 필드
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            input_types=[],
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
        # 기본 에이전트 입력들
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
        # System Prompt 입력 필드
        MultilineInput(
            name="system_prompt",
            display_name="Planner Instructions",
            info="A2A 플래너의 초기 지시사항과 컨텍스트",
            value="""당신은 A2A Discovery 서비스와 연동하여 작업 계획을 수립하는 A2A 플래너입니다.

주요 역할:
1. A2A Discovery 서비스에서 사용 가능한 에이전트 목록을 조회
2. 사용자 작업을 분석하고 작업 분할 계획 수립
3. 적절한 에이전트 매핑 및 실행 계획 제시

작업 계획 수립 시 다음 원칙을 준수하세요:
- 목표를 명확한 산출물 중심으로 분해 (30-90분 내 완료 가능한 크기)
- 각 태스크에 담당 에이전트/필요 스킬/입출력/완료 기준/의존성/추정 시간 명시
- 위상정렬 가능한 순서의 실행 계획 제시
- 리스크와 대안 포함, 병렬화 가능 구간 명시
- OUTPUT_SCHEMA 형식으로만 응답""",
            advanced=False,
        ),
        # Work Plan Prompt 입력 필드 (필수)
        HandleInput(
            name="work_plan_prompt",
            display_name="Work Plan Prompt (필수)",
            info="작업 계획 수립을 위한 프롬프트 템플릿 (Prompt Template 컴포넌트 연결 필수)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),
    ]
    
    outputs = [
        Output(name="response", display_name="Work Plan", method="build_agent")
    ]

    def _extract_user_text(self, input_value) -> str:
        """사용자 입력에서 실제 텍스트만 추출"""
        try:
            # Message 객체인 경우
            if hasattr(input_value, 'text'):
                return input_value.text
            elif hasattr(input_value, 'content'):
                return input_value.content
            
            # 문자열인 경우
            if isinstance(input_value, str):
                # JSON 형태인지 확인
                try:
                    import json
                    parsed = json.loads(input_value)
                    # Langflow Message 형태인 경우 text 필드 추출
                    if isinstance(parsed, dict) and 'text' in parsed:
                        return parsed['text']
                    # 일반 JSON인 경우 그대로 반환
                    return input_value
                except (json.JSONDecodeError, TypeError):
                    # JSON이 아닌 일반 문자열
                    return input_value
            
            # 기타 타입인 경우 문자열로 변환
            return str(input_value)
            
        except Exception as e:
            logger.warning(f"Error extracting user text: {e}")
            return str(input_value)

    async def _initialize_llm(self):
        """LLM 및 필수 속성 초기화"""
        try:
            # chat_history 초기화
            if not hasattr(self, 'chat_history'):
                self.chat_history = []
                logger.info("Initialized empty chat_history")
            
            # agent.py와 동일한 방식으로 LLM 생성/해결
            if not hasattr(self, 'agent_llm') or self.agent_llm is None:
                logger.warning("No agent_llm provided, LLM will not be available")
                self.llm = None
                return

            # 이미 객체면 그대로 사용
            if not isinstance(self.agent_llm, str):
                self.llm = self.agent_llm
                return

            # 문자열인 경우 provider로 간주하여 실제 모델 인스턴스 생성
            provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
            if not provider_info:
                logger.error(f"Invalid model provider: {self.agent_llm}")
                self.llm = None
                return

            component_class = provider_info.get("component_class")
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix", "")

            # provider 입력값을 현재 컴포넌트 속성에서 수집하여 모델 빌드
            model_kwargs = {}
            for input_ in inputs:
                attr_name = f"{prefix}{input_.name}"
                if hasattr(self, attr_name):
                    model_kwargs[input_.name] = getattr(self, attr_name)
            try:
                self.llm = component_class.set(**model_kwargs).build_model()
                logger.info("LLM instance built from provider configuration")
            except Exception as e:
                logger.error(f"Failed to build LLM instance: {e}")
                self.llm = None
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    async def get_a2a_agents(self) -> List[Dict[str, Any]]:
        """A2A Discovery 서비스에서 사용 가능한 에이전트 목록을 조회"""
        try:
            api_url = f"{self.a2a_api_base}/agents?status=active"
            logger.info(f"A2A Discovery : Fetching agents from {api_url}")
            
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
                "updated": {"type": "boolean", "description": "계획이 업데이트되었는지 여부"},
                "work_breakdown": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "태스크 ID (예: T1, T2)"},
                            "title": {"type": "string", "description": "태스크 제목"},
                            "description": {"type": "string", "description": "간략한 태스크 설명"},                            
                            "agent": {
                                "type": "object",
                                "properties": {
                                    "name": { "type": "string", "description": "담당 에이전트 ID 또는 이름" },
                                    "url": {"type": "string", "description": "담당 에이전트 URL" }
                                    }
                            },
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

        # 사용자 입력에서 실제 텍스트 추출
        user_task = self._extract_user_text(self.input_value)
        self.user_task = user_task  # 인스턴스 변수로 저장
        
        # LLM이 있는지 확인하고 없으면 기본 계획 반환
        if not hasattr(self, 'llm') or self.llm is None:
            logger.warning("LLM not available, using default work plan")
            return self.get_default_work_plan(user_task, available_agents)

        # LLM이 호출 가능한 객체인지 확인
        if not hasattr(self.llm, 'invoke') and not hasattr(self.llm, '__call__'):
            logger.warning("LLM is not a callable object, using default work plan")
            return self.get_default_work_plan(user_task, available_agents)

        try:
            # chat_history를 LangChain 형식으로 변환
            langchain_chat_history = []
            if hasattr(self, 'chat_history') and self.chat_history:
                for msg in self.chat_history:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if msg.type == "human":
                            langchain_chat_history.append(HumanMessage(content=msg.content))
                        elif msg.type == "ai":
                            langchain_chat_history.append(AIMessage(content=msg.content))
                        else:
                            langchain_chat_history.append(HumanMessage(content=str(msg.content)))

            # LLM 호출 방식 결정
            if hasattr(self.llm, 'invoke'):
                # LangChain 체인 사용
                chain = planning_prompt | self.llm | JsonOutputParser()
                result = await chain.ainvoke({
                    "user_task": user_task,
                    "available_agents": json.dumps(available_agents, ensure_ascii=False, indent=2),
                    "output_schema": json.dumps(output_schema, ensure_ascii=False, indent=2),
                    "chat_history": langchain_chat_history
                })
            else:
                # 직접 호출 후 JSON 파싱
                prompt_input = {
                    "user_task": user_task,
                    "available_agents": json.dumps(available_agents, ensure_ascii=False, indent=2),
                    "output_schema": json.dumps(output_schema, ensure_ascii=False, indent=2),
                    "chat_history": langchain_chat_history
                }
                formatted_prompt = planning_prompt.format(**prompt_input)
                llm_result = await self.llm(formatted_prompt)
                
                # JSON 파싱
                response_text = llm_result.content if hasattr(llm_result, 'content') else str(llm_result)
                result = json.loads(response_text)
            
            logger.info(f"Work plan created with {len(result.get('work_breakdown', []))} tasks")

            # 수립된 작업 제목들 출력
            self._log_work_breakdown(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating work plan with LLM: {e}")
            logger.info("Falling back to default work plan")
            return self.get_default_work_plan(user_task, available_agents)

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
            "updated": False,
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
                logger.info(f"  {i}. [{task.get('id', 'Unknown')}] {title}")
                if outputs:
                    logger.info(f"     출력: {', '.join(outputs)}")

    async def build_agent(self) -> Message:
        """A2A 플래너 실행 - 작업 계획만 수립하고 반환"""
        try:
            logger.info("A2A Planner starting...")
            
            # LLM 초기화
            await self._initialize_llm()
            
            # 1. A2A Discovery에서 에이전트 목록 조회
            available_agents = await self.get_a2a_agents()
            logger.info(f"A2A Discovery found {len(available_agents)} agents")
            
            # 2. 작업 계획 수립
            work_plan = await self.create_work_plan(available_agents)
            logger.info(f"Work plan created successfully with {len(work_plan.get('work_breakdown', []))} tasks")
            
            # 3. 작업 계획을 JSON 형태로 반환
            plan_json = json.dumps(work_plan, ensure_ascii=False, indent=2)
            
            return Message(
                text=plan_json,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )
            
        except Exception as e:
            error_msg = f"A2A Planner error: {str(e)}"
            logger.error(error_msg)
            return Message(
                text=error_msg,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """모든 필드의 input_types 업데이트(AgentComponent와 동일한 정책)"""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
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
                # AgentComponent와 동일: agent_llm의 input_types는 비워 외부 컴포넌트 강제 안 함
                build_config["agent_llm"]["input_types"] = []

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
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "handle_parsing_errors",
                "verbose",
                "work_plan_prompt",
            ]
            
            # 누락된 필수 키들을 기본값으로 설정
            if "input_value" not in build_config:
                build_config["input_value"] = {"value": "", "input_types": ["Message"]}
            if "agent_description" not in build_config:
                build_config["agent_description"] = {"value": "A helpful assistant with access to the following tools:"}
            if "work_plan_prompt" not in build_config:
                build_config["work_plan_prompt"] = {"value": None, "input_types": ["Message"]}
            
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})