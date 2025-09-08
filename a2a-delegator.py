import json
import asyncio
from typing import List, Dict, Any, Optional
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
from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from langflow.custom.custom_component.component import _get_component_toolkit
from langflow.custom.utils import update_component_build_config
from langflow.io import BoolInput, DropdownInput, HandleInput, IntInput, MultilineInput, Output, StrInput
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_AI


def create_prompt_from_component(prompt_component, default_system_message: str = "") -> ChatPromptTemplate:
    """프롬프트 컴포넌트에서 ChatPromptTemplate을 생성하는 헬퍼 함수"""
    if not prompt_component:
        logger.warning("prompt_component is None or empty")
        return None
        
    try:
        if hasattr(prompt_component, 'template'):
            template_content = prompt_component.template
        elif hasattr(prompt_component, 'text'):
            template_content = prompt_component.text
        else:
            logger.warning("Prompt component format not recognized")
            return None
        
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


class A2ADelegatorComponent(ToolCallingAgentComponent):
    display_name: str = "A2A Delegator"
    description: str = "A2A 실행 계획을 받아 순차적으로 작업을 실행하고 PDC 사이클을 관리하는 위임 에이전트"
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "A2A Delegator"

    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Model Provider",
            info="The provider of the language model that the agent will use to generate responses.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=False,
            input_types=[],
            options_metadata=[MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST] + [{"icon": "brain"}],
        ),
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],

        HandleInput(
            name="execution_plan",
            display_name="Execution Plan",
            info="A2A Planner에서 생성된 실행 계획 (JSON Message)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),

        StrInput(
            name="a2a_api_base",
            display_name="A2A API Base URL",
            info="A2A Discovery 서비스의 API 기본 URL",
            value="http://localhost:8000",
            advanced=False,
        ),





        IntInput(
            name="max_iterations",
            display_name="Max Iterations",
            info="최대 PDC 사이클 반복 횟수",
            value=10,
            advanced=True,
        ),

        IntInput(
            name="task_timeout",
            display_name="Task Timeout (seconds)",
            info="각 작업의 최대 실행 시간(초)",
            value=300,
            advanced=True,
        ),



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
            value=True,
            info="Whether to run in verbose mode.",
            advanced=True,
        ),

        MultilineInput(
            name="system_prompt",
            display_name="Delegator Instructions", 
            info="A2A Delegator의 시스템 지시사항",
            value="""당신은 A2A Discovery 서비스와 연동하여 실행 계획에 따라 작업을 위임하고 관리하는 A2A Delegator입니다.

주요 역할:
1. A2A Planner가 생성한 실행 계획(execution_plan)을 분석
2. work_breakdown에서 의존성을 고려하여 다음 실행 가능한 작업 하나를 선택
3. 각 작업 실행 전 산출물이 이미 등록되었는지 LLM으로 검증
4. 작업 실행 후 결과를 출력 포트로 A2A Critic 컴포넌트에 전달
5. Critic이 평가 결과를 A2A Planner에 피드백하여 계획 업데이트
6. 순환 구조: Planner → Delegator → Critic → Planner → Delegator...

실행 원칙:
- 의존성 관계를 준수하여 순차적으로 한 번에 하나씩 작업 실행
- 산출물이 이미 존재하면 작업 스킵하고 다음 작업으로 이동
- 각 작업의 결과를 추적하여 전체 진행 상황 관리
- 오류 발생 시 적절한 대응 및 로깅""",
            advanced=False,
        ),

        BoolInput(
            name="enable_output_validation",
            display_name="Enable Output Validation",
            info="execution_history를 기반으로 산출물 중복 검증을 활성화할지 여부",
            value=True,
            advanced=False,
        ),

        *LCToolsAgentComponent._base_inputs
    ]
    
    outputs = [
        Output(name="execution_history_out", display_name="Execution History", group_outputs=True, method="execution_history_output"),
        Output(name="response", display_name="Execution Result", group_outputs=True, method="execute_plan")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_outputs = {}
        self.current_iteration = 0
        self.execution_history = []
        self.completed_tasks = set()

    async def _initialize_llm(self):
        """LLM 초기화 - agent.py와 동일한 방식"""
        try:
            if not hasattr(self, 'chat_history'):
                self.chat_history = []
                logger.info("Initialized empty chat_history")
            
            # agent_llm이 설정되어 있는지 확인
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

    def parse_execution_plan(self, plan_message: Message) -> Dict[str, Any]:
        """실행 계획 메시지를 파싱"""
        try:
            if hasattr(plan_message, 'text'):
                plan_text = plan_message.text
            else:
                plan_text = str(plan_message)
            
            execution_plan = json.loads(plan_text)
            logger.info(f"Parsed execution plan with {len(execution_plan.get('work_breakdown', []))} tasks")
            return execution_plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse execution plan JSON: {e}")
            raise ValueError(f"Invalid execution plan format: {e}")
        except Exception as e:
            logger.error(f"Error parsing execution plan: {e}")
            raise

    def get_next_executable_task(self, work_breakdown: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """의존성을 고려하여 다음에 실행 가능한 작업 하나를 반환 (순차 실행)"""
        for task in work_breakdown:
            task_id = task.get('id')
            
            if task_id in self.completed_tasks:
                continue
                
            dependencies = task.get('dependencies', [])
            dependencies_met = all(dep_id in self.completed_tasks for dep_id in dependencies)
            
            if dependencies_met:
                return task
        
        return None

    async def validate_output_exists(self, task: Dict[str, Any]) -> bool:
        """execution_history를 기반으로 작업의 산출물이 이미 완료되었는지 검증"""
        try:
            # 검증 기능이 비활성화된 경우
            if not getattr(self, 'enable_output_validation', True):
                logger.info(f"Output validation disabled, proceeding with execution for task '{task.get('title')}'")
                return False
                
            task_outputs = task.get('outputs', [])
            if not task_outputs:
                # 산출물이 정의되지 않은 작업은 중복 실행 허용
                logger.info(f"Task '{task.get('title')}' has no defined outputs, proceeding with execution")
                return False
            
            # execution_history에서 완료된 산출물들 수집
            completed_outputs = set()
            for history_item in self.execution_history:
                if history_item.get('outputs_completed', False):
                    completed_outputs.update(history_item.get('task_outputs', []))
            
            # 현재 작업의 모든 산출물이 이미 완료되었는지 확인
            required_outputs = set(task_outputs)
            missing_outputs = required_outputs - completed_outputs
            
            if not missing_outputs:
                logger.info(f"Task '{task.get('title')}' outputs {task_outputs} already completed, skipping execution")
                return True
            else:
                logger.info(f"Task '{task.get('title')}' missing outputs {list(missing_outputs)}, proceeding with execution")
            return False
            
        except Exception as e:
            logger.error(f"Error validating output existence: {e}")
            logger.info(f"Falling back to assume output doesn't exist for task '{task.get('title')}'")
            return False

    async def get_agent_info(self, agent_info: str) -> Optional[Dict[str, Any]]:
        """A2A Discovery에서 에이전트 정보 조회"""
        try:
            agent_name = agent_info.get("name","Unknown")      
            agent_url = agent_info.get("url","Unknown").replace("0.0.0.0","127.0.0.1")
            agent_url = agent_url.rstrip("/")
            api_url = f"{agent_url}/.well-known/agent-card.json"
            logger.info(f"Getting agent : {api_url}")
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        agent_info = await response.json()
                        logger.info(f"Found agent info for {agent_name}")
                        return agent_info
                    elif response.status == 404:
                        logger.info(f"Agent {agent_name} not found in A2A Discovery (404)")
                        return None
                    else:
                        logger.warning(f"Agent {agent_name} lookup failed with status {response.status}")
                        return None
                        
        except aiohttp.ClientConnectorError as e:
            logger.info(f"Failed to connect to A2A Discovery service at {self.a2a_api_base}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting agent info for {agent_name}: {e}")
            return None

    async def delegate_to_agent(self, task: Dict[str, Any], agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """외부 에이전트에 작업 위임"""
        task_id = task.get('id')
        task_title = task.get('title', 'Unknown Task')
        
        logger.info(f"Delegating task {task_id} to agent: {agent_info.get('name', 'Unknown')}")
        
        try:
            agent_url = agent_info.get('url', agent_info.get('endpoint'))
            if not agent_url:
                raise ValueError("Agent URL not found")
            
            # verbose 모드에 따라 텍스트 앞부분 결정
            verbose_prefix = f"**응답은 주어진 task의 outputs에 해당하는 결과만 다음 json 으로 변환하여 반환해주세요.**\n반환형식\n{{'output 요소':'output 값', ... }}\n\n" if not self.verbose else ""
            
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
                                "text": f"{verbose_prefix}Task: {task['title']}\nDescription: {task.get('description', '')}\nInputs: {', '.join(task.get('inputs', []))}\nExpected Outputs: {', '.join(task.get('outputs', []))}"
                            }
                        ]
                    },
                    "stream": False  # 스트리밍 비활성화
                }
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "A2A-Agent-Component/1.0"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    agent_url,
                    json=a2a_rpc_payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        task_result = await response.json()
                        logger.info(f"Task {task_id} delegated successfully")
                        extracted_data = self.extract_text_from_agent_response(task_result)

                        if not extracted_data:
                            raise Exception("Agent response could not be parsed")

                        response_text = extracted_data["response_text"] 
                        if not self.verbose and isinstance(response_text, str):
                            response_text = self._parse_json_from_text(response_text)
                        
                        return {
                            "task_id": task["id"],
                            "agent_id": agent_info.get('name', 'Unknown'),
                            "status": "completed",
                            "timestamp": "2024-01-01T00:00:00Z",
                            "session_id": extracted_data["session_id"],
                            "input_text": extracted_data["input_text"],
                            "response_text": response_text
                        }
                    else:
                        error_text = await response.text()
                        raise Exception(f"Agent returned status {response.status}: {error_text}")
                        
        except Exception as e:
            logger.error(f"Error delegating task {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "execution_time": 0
            }
    def extract_text_from_agent_response(self, agent_response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """A2A 에이전트 응답에서 실제 텍스트를 추출하는 헬퍼 함수"""
        try:
            # agent_response가 이미 dict인 경우
            if isinstance(agent_response, dict):
                obj = agent_response
            # agent_response가 문자열인 경우 JSON 파싱
            elif isinstance(agent_response, str):
                if agent_response.strip().startswith('{'):
                    obj = json.loads(agent_response)
                else:
                    logger.warning("Agent response is not a valid JSON string")
                    return None
            else:
                logger.warning(f"Unexpected agent_response type: {type(agent_response)}")
                return None
            
            # JSON 구조에서 중첩된 JSON 문자열 파싱
            obj = self._parse_nested_json_strings(obj)
            
            # parts 필드들 찾기
            parts = self.find_parts(obj)
            
            if len(parts) == 0:
                logger.warning("No message fields found in agent response")
                return None

            #logger.info(f"agent_response parts: {parts[0]}") 
            
            # 안전한 데이터 추출
            try:
                session_id = self._extract_session_id(parts)
                input_text = self._extract_input_text(parts)
                response_text = self._extract_response_text(parts)                
            except Exception as e:
                logger.error(f"Error extracting data from parts: {e}")
                logger.error(f"Parts structure: {parts}")
                return None 
            
            if not response_text:
                logger.warning("No meaningful text content found in messages")
                return None
               
            return {
                "session_id": session_id,
                "input_text": input_text,
                "response_text": response_text
            }
        except Exception as e:
            logger.error(f"Error extracting text from agent response: {e}")
            return None
      
    def _parse_nested_json_strings(self, obj):
        """중첩된 JSON 문자열을 재귀적으로 파싱"""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if isinstance(value, str) and value.strip().startswith('{'):
                    try:
                        result[key] = self._parse_nested_json_strings(json.loads(value))
                    except json.JSONDecodeError:
                        result[key] = value
                else:
                    result[key] = self._parse_nested_json_strings(value)
            return result
        elif isinstance(obj, list):
            return [self._parse_nested_json_strings(item) for item in obj]
        else:
            return obj

    def _extract_input_text(self, parts: list) -> str:
        """parts에서 input_value를 재귀적으로 추출"""
        if not parts:
            return "unknown"
        
        result = self._find_value_recursive(parts, "input_value", "unknown")
        return result if result != "unknown" else "unknown"

    def _extract_response_text(self, parts: list) -> str:
        """parts에서 최종 텍스트를 재귀적으로 추출"""
        if not parts:
            return None
        
        # message.data.text 경로를 따라 찾기
        result = self._find_value_recursive(parts, "data", None)
        
        text = ""
        if result:
            try:
                text = result.get("text") if isinstance(result, dict) else ""
            except Exception:
                text = ""

        return text

    def _parse_json_from_text(self, text: str) -> Any:
        """텍스트에서 JSON 구성 정보만 추려서 JSON으로 변환하는 재사용 가능한 함수"""
        import re
        
        # JSON 객체 패턴 찾기 (중괄호로 둘러싸인 부분)
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from text: {json_str}")
                return {"error": "JSON parsing failed", "raw_text": text}
        else:
            logger.warning(f"No JSON pattern found in text: {text}")
            return {"error": "No JSON pattern found", "raw_text": text}
        
    def find_parts(self, obj: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        JSON(dict/list) 구조 내부를 재귀적으로 탐색하여 
        'parts' 필드로 지정된 내용을 모두 찾아 리스트로 반환하는 함수
        """
        results = []

        if isinstance(obj, dict):
            # 현재 dict에 'parts' 키가 있으면 추가
            if "parts" in obj:
                parts_content = obj["parts"]
                # parts가 이미 리스트이면 그대로 추가, 아니면 리스트로 변환
                if isinstance(parts_content, list):
                    results.extend(parts_content)
                else:
                    results.append(parts_content)

            # 재귀적으로 모든 값들을 탐색
            for value in obj.values():
                results.extend(self.find_parts(value))

        elif isinstance(obj, list):
            # 리스트의 각 항목을 재귀적으로 탐색
            for item in obj:
                results.extend(self.find_parts(item))

        return results

    def _find_value_recursive(self, obj: Any, target_key: str, default_value: Any = None) -> Any:
        """
        중첩된 dict/list 구조에서 특정 키의 값을 재귀적으로 찾는 함수
        
        Args:
            obj: 탐색할 객체 (dict, list, 또는 기타)
            target_key: 찾을 키 이름
            default_value: 찾지 못했을 때 반환할 기본값
            
        Returns:
            찾은 값 또는 기본값
        """
        if isinstance(obj, dict):
            # 현재 dict에 타겟 키가 있으면 반환
            if target_key in obj:
                return obj[target_key]
            
            # 모든 값들을 재귀적으로 탐색
            for value in obj.values():
                result = self._find_value_recursive(value, target_key, default_value)
                if result is not default_value:
                    return result
                    
        elif isinstance(obj, list):
            # 리스트의 각 요소를 순서대로 탐색
            for item in obj:
                result = self._find_value_recursive(item, target_key, default_value)
                if result is not default_value:
                    return result
        
        return default_value

    def _extract_session_id(self, parts: list) -> str:
        """parts에서 session_id를 재귀적으로 추출"""
        if not parts:
            return "unknown"
        
        result = self._find_value_recursive(parts, "session_id", "unknown")
        return result if result != "unknown" else "unknown"

    def prepare_critic_data(self, task_result: Dict[str, Any], 
                           current_plan: Dict[str, Any]) -> Dict[str, Any]:
        """A2A Critic에 전달할 데이터 준비"""
        logger.info("Preparing data for A2A Critic")
        
        try:
            # task_result에서 response_text 안전하게 처리
            response_text = task_result.get("response_text", "{}")
            if isinstance(response_text, str):
                try:
                    parsed_response = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse response_text as JSON: {response_text}")
                    parsed_response = {"raw_text": response_text}
            else:
                parsed_response = response_text if response_text else {}
            
            critique_data = {
                "execution_results": [task_result],
                "current_plan": current_plan,
                "task_outputs": self.task_outputs,
                "task_result": parsed_response,
                "completed_tasks": list(self.completed_tasks),
                "total_tasks": len(current_plan.get('work_breakdown', [])),
                "progress": f"{len(self.completed_tasks)}/{len(current_plan.get('work_breakdown', []))}"
            }
            
            logger.info(f"Critic data structure prepared successfully")
            return critique_data
            
        except Exception as e:
            logger.error(f"Error preparing critic data: {e}")
            # 안전한 기본값 반환
            return {
                "execution_results": [task_result],
                "current_plan": current_plan,
                "task_outputs": self.task_outputs,
                "task_result": {},
                "completed_tasks": list(self.completed_tasks),
                "total_tasks": len(current_plan.get('work_breakdown', [])),
                "progress": f"{len(self.completed_tasks)}/{len(current_plan.get('work_breakdown', []))}",
                "error": f"Critic data preparation failed: {str(e)}"
            }




    def all_tasks_completed(self, work_breakdown: List[Dict[str, Any]]) -> bool:
        """모든 작업이 완료되었는지 확인"""
        all_task_ids = {task.get('id') for task in work_breakdown}
        return all_task_ids.issubset(self.completed_tasks)

    async def execute_plan(self) -> Message:
        """실행 계획을 받아 PDC 사이클을 실행"""
        try:
            logger.info("A2A Delegator starting execution...")
            
            await self._initialize_llm()
            
            execution_plan = self.parse_execution_plan(self.execution_plan)
            work_breakdown = execution_plan.get('work_breakdown', [])
            
            if not work_breakdown:
                return Message(
                    text="No tasks found in execution plan",
                    sender=MESSAGE_SENDER_AI,
                    sender_name=MESSAGE_SENDER_NAME_AI,
                )
            
            logger.info(f"Starting task execution with {len(work_breakdown)} tasks")
            
            # 현재 실행 계획에서 다음 실행 가능한 작업 하나 찾기
            next_task = self.get_next_executable_task(work_breakdown)
            
            if not next_task:
                if self.all_tasks_completed(work_breakdown):
                    logger.info("All tasks completed successfully!")
                    final_result = {
                        "status": "completed",
                        "completed_tasks": list(self.completed_tasks),
                        "total_tasks": len(work_breakdown),
                        "task_outputs": self.task_outputs,
                        "execution_summary": self.execution_history
                    }
                else:
                    logger.warning("No executable tasks found but not all tasks completed")
                    final_result = {
                        "status": "deadlock",
                        "completed_tasks": list(self.completed_tasks),
                        "total_tasks": len(work_breakdown),
                        "task_outputs": self.task_outputs,
                        "execution_summary": self.execution_history
                    }
            else:
                # 다음 작업 실행
                logger.info(f"Executing next task: {next_task.get('id')} - {next_task.get('title')}")
                task_result = await self.execute_task(next_task)
                self.execution_history.append(task_result)

                self.execution_history_output()
                
                # A2A Critic에 전달할 데이터 준비
                critic_data = self.prepare_critic_data(task_result, execution_plan)
                logger.info(f"Critic data prepared: {type(critic_data)} with keys: {list(critic_data.keys()) if isinstance(critic_data, dict) else 'Not a dict'}")
                final_result = {
                    "status": "task_completed",
                    "executed_task": task_result,
                    "critic_data": critic_data,
                    "completed_tasks": list(self.completed_tasks),
                    "total_tasks": len(work_breakdown),
                    "task_outputs": self.task_outputs,
                    "remaining_tasks": len(work_breakdown) - len(self.completed_tasks)
                }
            
            result_json = json.dumps(final_result, ensure_ascii=False, indent=2)
            #logger.info(f"---------------Critic data: {critic_data}")
            return Message(
                text=result_json,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )
            
        except Exception as e:
            error_msg = f"A2A Delegator execution error: {str(e)}"
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

    async def update_build_config(self, build_config: dotdict, field_value: str, field_name: str | None = None) -> dotdict:
        try:
            if field_name in ("agent_llm",):
                # 값만 갱신 (shape 불변)
                if "agent_llm" in build_config:
                    build_config["agent_llm"]["value"] = field_value

            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
                        # 내부적으로 model_name만 보강 (필드 추가/삭제 최소화)
                    build_config = await update_component_build_config(
                        component_class, build_config, field_value, "model_name"
                    )

                    # 기존 코드에서 대규모 delete/update 하던 부분은 주석/제거:
                    # self.delete_fields(...), fields_to_add/fields_to_delete 대량 갱신 등
                    # → 연결 유실을 유발하므로 지양

                # 필수키 확인도 "raise" 대신 보강만
                default_keys = [
                    "code", "_type", "agent_llm", "execution_plan", "a2a_api_base",
                    "max_iterations", "task_timeout",
                    "enable_output_validation", "handle_parsing_errors", "verbose", "system_prompt",
                ]
                for k in default_keys:
                    if k not in build_config:
                        # 없는 경우 기본 스펙만 채워 넣고 넘어감
                        # (필요시 최소 구조만 추가)
                        pass

            # input_types 보정은 그대로 두되, 값만 수정
            build_config = self.update_input_types(build_config)

            return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})
        except Exception as e:
            logger.warning(f"update_build_config soft-failed: {e}")
            # 실패해도 기존 스키마 그대로 반환
            return build_config

    def execution_history_output(self) -> Message:
        """실행 이력을 반환하는 메서드"""
        try:
            history_data = {
                "execution_history": self.execution_history,
                "completed_tasks": list(self.completed_tasks),
                "task_outputs": self.task_outputs,
                "total_executions": len(self.execution_history),
                "completion_rate": len(self.completed_tasks) / max(len(self.execution_history), 1) * 100,
                "last_updated": "현재 시점"
            }
            
            history_json = json.dumps(history_data, ensure_ascii=False, indent=2)
            
            return Message(
                text=f"{history_json}",
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )
            
        except Exception as e:
            error_msg = f"Error retrieving execution history: {str(e)}"
            logger.error(error_msg)
            return Message(
                text=error_msg,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )

    def delete_fields(self, build_config: dotdict, fields: dict | list[str]) -> None:
        """Delete specified fields from build_config."""
        for field in fields:
            build_config.pop(field, None)

    def get_llm(self):
        """a2a-agent.py의 get_llm 메서드 추가"""
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
        """a2a-agent.py의 _build_llm_model 메서드 추가"""
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

    def set_component_params(self, component):
        """a2a-agent.py의 set_component_params 메서드 추가"""
        provider_info = MODEL_PROVIDERS_DICT.get(self.agent_llm)
        if provider_info:
            inputs = provider_info.get("inputs")
            prefix = provider_info.get("prefix")
            model_kwargs = {input_.name: getattr(self, f"{prefix}{input_.name}") for input_ in inputs}

            return component.set(**model_kwargs)
        return component
