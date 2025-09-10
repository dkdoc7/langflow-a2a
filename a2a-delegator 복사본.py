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
from langflow.custom.utils import update_component_build_config
from langflow.io import HandleInput, BoolInput, DropdownInput, IntInput, MultilineInput, Output
from langflow.logging import logger
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message

from langflow.utils.constants import MESSAGE_SENDER_AI, MESSAGE_SENDER_NAME_AI
from langchain_core.messages import HumanMessage
import json
from typing import Any, Dict, List, Optional
import aiohttp

def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


MODEL_PROVIDERS_LIST = ["Anthropic", "Google Generative AI", "Groq", "OpenAI"]


class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "A2A Delegator"
    description: str = "Delegate a task to an agent."
    icon = "bot"
    beta = False
    name = "Agent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
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
        *MODEL_PROVIDERS_DICT["OpenAI"]["inputs"],
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
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

        #----------------------------------------------------

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
1. A2A Planner가 생성한 실행 계획(input_value)을 분석
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
        
        #----------------------------------------------------









        *LCToolsAgentComponent._base_inputs,
        # removed memory inputs from agent component
        # *memory_inputs,

        
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [
        Output(name="response", display_name="Execution Result", method="execute_plan")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Execution state for plan-delegation mode (optional)
        self.task_outputs: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.completed_tasks: set[str] = set()

    def build_config(self):
        cfg = super().build_config()
        cfg["input_value"]["display_name"] ='dkdoc'
        return cfg

    

    async def execute_plan(self) -> Message:
        try:
            # Get LLM model and validate
            llm_model, display_name = self.get_llm()
            if llm_model is None:
                msg = "No language model selected. Please choose a model to proceed."
                raise ValueError(msg)
            self.model_name = get_model_name(llm_model, display_name=display_name)

            # Get memory data
            self.chat_history = await self.get_memory_data()
            if isinstance(self.chat_history, Message):
                self.chat_history = [self.chat_history]

            # Add current date tool if enabled
            if self.add_current_date_tool:
                if not isinstance(self.tools, list):  # type: ignore[has-type]
                    self.tools = []
                current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)
                if not isinstance(current_date_tool, StructuredTool):
                    msg = "CurrentDateComponent must be converted to a StructuredTool"
                    raise TypeError(msg)
                self.tools.append(current_date_tool)
            # note the tools are not required to run the agent, hence the validation removed.

            # Set up and run agent
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )




            #----------------------------------------------------
            input_value = self.parse_input_value(self.input_value)
            work_breakdown = input_value.get('work_breakdown', [])
            
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

                try:                               
                    task_id = next_task.get('id')

                    # 산출물이 이미 존재하면 작업 스킵
                    if await self.validate_output_exists(next_task):
                        result = {
                            "task_id": task_id,
                            "status": "skipped",
                            "reason": "Output already exists",
                            "outputs": next_task.get('outputs', []),
                            "task_outputs": next_task.get('outputs', []),
                            "outputs_completed": True,
                            "execution_time": 0
                        }
                        self.completed_tasks.add(task_id)
                        task_result = result

                    task_title = next_task.get('title', 'Unknown Task')
                    agent = next_task.get('agent', None)
                    agent_name = agent.get("name","Unknown") if agent else "Unknown"
                    task_result = None

                    # agent가 지정되었고 'self'가 아니라면 작업 위임
                    if agent and agent_name != 'self':
                        task_result = await self.delegate_to_agent(next_task, agent)
                    else:
                        agent = self.create_agent_runnable()
                        task_result = await self.run_agent(agent)

                    # 결과 정규화: Message -> dict
                    if isinstance(task_result, Message):
                        result_text = task_result.text if hasattr(task_result, "text") else str(task_result)
                        task_result = {
                            "task_id": task_id,
                            "status": "completed",
                            "task_description": task_title,
                            "result": result_text,
                        }

                    # 산출물 저장
                    for output in next_task.get('outputs', []):
                        self.task_outputs[output] = task_result.get('result', 'Task completed') if isinstance(task_result, dict) else 'Task completed'
                    
                    # 실행 결과에 산출물 정보 추가
                    if isinstance(task_result, dict):
                        task_result['task_outputs'] = next_task.get('outputs', [])
                        task_result['outputs_completed'] = True if task_result.get('status') == 'completed' else False
                    
                    self.completed_tasks.add(task_id)
                    logger.info(f"Task {task_id} completed successfully")

                except Exception as e:
                        logger.error(f"Error executing task {task_id}: {e}")
                        task_result = {
                            "task_id": task_id,
                            "status": "failed",
                            "error": str(e),
                            "task_outputs": next_task.get('outputs', []),
                            "outputs_completed": False,
                            "execution_time": 0
                        }

                self.execution_history.append(task_result)

                self.execution_history_output()
                
                # A2A Critic에 전달할 데이터 준비
                critic_data = self.prepare_critic_data(task_result, input_value)
                '''final_result = {
                    "status": "task_completed",
                    "executed_task": task_result,
                    "critic_data": critic_data,
                    "completed_tasks": list(self.completed_tasks),
                    "total_tasks": len(work_breakdown),
                    "task_outputs": self.task_outputs,
                    "remaining_tasks": len(work_breakdown) - len(self.completed_tasks)
                }'''
            #----------------------------------------------------


            result_json = json.dumps(critic_data, ensure_ascii=False, indent=2)
            return Message(
                text=result_json,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )
            
        except Exception as e:
            error_msg = f"Agent plan execution error: {str(e)}"
            logger.error(error_msg)
            return Message(
                text=error_msg,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )

    def parse_input_value(self, plan_message: Message) -> Dict[str, Any]:
        """실행 계획 메시지를 파싱"""
        try:
            if hasattr(plan_message, 'text'):
                plan_text = plan_message.text
            else:
                plan_text = str(plan_message)
            
            input_value = json.loads(plan_text)
            return input_value
            
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


    async def execute_builtin_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """내장 기능으로 작업 실행 - a2a-agent.py의 execute_task_directly 방식 사용"""
        task_id = task.get('id')
        task_title = task.get('title', 'Unknown Task')
             
        # a2a-agent.py의 execute_task_directly 로직 적용
        try:
            # Tool 힌트가 여러 도구와 충돌하지 않도록 항상 비활성화
            if hasattr(self, "tool_name"):
                self.tool_name = ""
            if hasattr(self, "tool_description"):
                self.tool_description = ""

            # 작업 단위 입력 프롬프트 구성 (agent.py의 input_value와 유사하게 단일 문자열)
            inputs = ", ".join(task.get("inputs", []))
            outputs = ", ".join(task.get("outputs", []))
            description = task.get("description", "")
            task_prompt = f"Task: {task_title}\nDescription: {description}\nInputs: {inputs}\nExpected Outputs: {outputs}".strip()

            # 1) Tool이 연결되어 있으면 Tool 우선 사용
            has_tools = hasattr(self, "tools") and isinstance(self.tools, list) and len(self.tools) > 0
            if has_tools:
                logger.info(f"Self-Executing with Tool : {self.tools}")
                try:
                    # 여러 도구 사용 허용하되, 단일-툴 힌트는 비활성화하여 검증 충돌 방지
                    if hasattr(self, "tool_name"):
                        self.tool_name = ""
                    if hasattr(self, "tool_description"):
                        self.tool_description = ""

                    # 기존 도구를 직접 사용하여 Agent 실행 (복수 도구 사용 허용)
                    self.set(
                        llm=self.llm,
                        tools=self.tools,
                        chat_history=getattr(self, "chat_history", []),
                        input_value=task_prompt,
                        system_prompt=getattr(self, "system_prompt", ""),
                    )
                    agent = self.create_agent_runnable()
                    agent_result_msg: Message = await self.run_agent(agent)
                    result_text = agent_result_msg.text if hasattr(agent_result_msg, "text") else str(agent_result_msg)

                    return {
                        "task_id": task_id,
                        "execution_method": "tools_agent",
                        "status": "completed",
                        "task_description": task_title,
                        "user_input": task_prompt,
                        "result": result_text,
                    }
                except Exception as tool_err:
                    logger.warning(f"Tool execution failed, falling back to LLM: {tool_err}")
            else:
                logger.info("Self-Executing with LLM (No Tools)")

            # 2) Tool이 없거나 실패한 경우: LLM 기반 일반 처리
            return await self.task_with_LLM(task, task_prompt)
                
        except Exception as e:
            logger.error(f"Error executing builtin task {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "execution_time": 0
            }

    async def task_with_LLM(self, task: Dict[str, Any], user_input: str) -> Dict[str, Any]:
        
        # 기본적인 작업 처리
        task_description = task.get("title", "LLM-Based Task")
        
        # LLM이 있는 경우 LLM을 사용
        if hasattr(self, 'llm') and self.llm is not None:
            try:

                verbose_prefix = f"**응답은 주어진 task의 outputs에 해당하는 결과만 다음 json 으로 변환하여 반환해주세요.**\n반환형식\n{{'output 요소':'output 값', ... }}\n\n" if not self.verbose else ""

                prompt = f"""{verbose_prefix}
                
                다음 작업을 수행해주세요:

작업: {task_description}
사용자 입력: {user_input}

적절한 방식으로 작업을 완료하고 결과를 제공해주세요."""
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                result_content = response.content if hasattr(response, 'content') else str(response)
                
                # verbose 모드가 아닌 경우 JSON 파싱 적용
                if not self.verbose:
                    parsed_result = self._parse_json_from_text(result_content)
                    if isinstance(parsed_result, dict) and "error" not in parsed_result:
                        result_content = parsed_result
                
                logger.info(f"Task result: {result_content}")
                
                return {
                    "task_id": task["id"],
                    "execution_method": "llm_general",
                    "status": "completed",
                    "task_description": task_description,
                    "user_input": user_input,
                    "result": f"🔧 작업: {task_description}\n📝 입력: {user_input}\n\n✅ 결과:\n{result_content}"
                }
            except Exception as e:
                logger.error(f"LLM Task Execution Error: {e}")
        
        # LLM이 없거나 오류가 발생한 경우 기본 처리
        result_content = f"'{task_description}' Task with '{user_input}' input has been processed. For more precise processing, an LLM model or dedicated tools are required."
        logger.info(f"Basic general task result: {result_content}")
        
        return {
            "task_id": task["id"],
            "execution_method": "basic_general",
            "status": "completed",
            "task_description": task_description,
            "user_input": user_input,
            "result": f"🔧 작업: {task_description}\n📝 입력: {user_input}\n\n✅ 기본 처리 결과:\n{result_content}"
        }

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
                        result = await response.json()
                        logger.info(f"Task {task_id} delegated successfully")
                        extracted_data = self.extract_text_from_agent_response(result)

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
        if  result:
            try:
                text=result.get("text") if isinstance(result, dict) else ""
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
                #"current_plan": current_plan,
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