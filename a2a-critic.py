import json
import asyncio
from typing import List, Dict, Any, Optional
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


class A2ACriticComponent(ToolCallingAgentComponent):
    display_name: str = "A2A Critic"
    description: str = "A2A Delegator의 실행 결과를 평가하고 Planner에게 피드백을 제공하는 비평 에이전트"
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "A2A Critic"

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

        # Delegator로부터 실행 결과 입력
        HandleInput(
            name="execution_data",
            display_name="Execution Data",
            info="A2A Delegator에서 전달된 실행 결과 데이터 (JSON Message)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),

        # 평가를 위한 프롬프트
        HandleInput(
            name="critique_prompt",
            display_name="Critique Prompt (필수)",
            info="실행 결과 평가를 위한 프롬프트 템플릿 (Prompt Template 컴포넌트 연결 필수)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),

        # 계획 업데이트를 위한 프롬프트
        HandleInput(
            name="plan_update_prompt",
            display_name="Plan Update Prompt (필수)",
            info="계획 업데이트 제안을 위한 프롬프트 템플릿 (Prompt Template 컴포넌트 연결 필수)",
            advanced=False,
            input_types=["Message"],
            required=True,
        ),

        # 평가 기준 설정
        IntInput(
            name="quality_threshold",
            display_name="Quality Threshold (%)",
            info="작업 품질 임계값 (이 값 이하면 재실행 제안)",
            value=80,
            advanced=True,
        ),

        BoolInput(
            name="suggest_improvements",
            display_name="Suggest Improvements",
            info="개선사항 제안 활성화 여부",
            value=True,
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
            value=True,
            info="Whether to run in verbose mode.",
            advanced=True,
        ),

        # System Prompt
        MultilineInput(
            name="system_prompt",
            display_name="Critic Instructions",
            info="A2A Critic의 시스템 지시사항",
            value="""당신은 A2A Discovery 시스템에서 작업 실행 결과를 평가하고 계획 개선을 제안하는 A2A Critic입니다.

주요 역할:
1. A2A Delegator가 실행한 작업 결과를 종합적으로 평가
2. 작업 품질, 완성도, 효율성을 다각도로 분석
3. 실행 계획의 문제점과 개선점 식별
4. A2A Planner에게 구체적이고 실행 가능한 피드백 제공
5. 전체 프로젝트 진행 상황을 추적하고 리스크 평가

평가 기준:
- 작업 완성도: 요구사항 충족 여부, 산출물 품질
- 실행 효율성: 시간, 자원 사용의 적절성
- 의존성 관리: 선후 관계 및 병목 지점 분석
- 리스크 요소: 잠재적 문제점과 대응 방안
- 개선 가능성: 더 나은 접근 방법 제안

피드백 원칙:
- 구체적이고 실행 가능한 개선안 제시
- 긍정적 측면과 개선 필요 사항 균형있게 평가
- 전체 프로젝트 맥락에서 개별 작업 평가
- 명확한 우선순위와 다음 단계 제안""",
            advanced=False,
        ),
    ]
    
    outputs = [
        Output(name="response", display_name="Critique Result", method="evaluate_execution")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_history = []  # 평가 이력

    async def _initialize_llm(self):
        """LLM 초기화"""
        try:
            if not hasattr(self, 'chat_history'):
                self.chat_history = []
                logger.info("Initialized empty chat_history")
            
            if hasattr(self, 'llm') and self.llm is not None:
                logger.info("LLM already initialized")
                return
                
            if hasattr(self, 'agent_llm') and self.agent_llm is not None:
                logger.info(f"Initializing LLM with agent_llm: {type(self.agent_llm)}")
                self.llm = self.agent_llm
            else:
                logger.warning("No agent_llm provided, LLM will not be available")
                self.llm = None
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None

    def parse_execution_data(self, execution_message: Message) -> Dict[str, Any]:
        """실행 데이터 메시지를 파싱"""
        try:
            if hasattr(execution_message, 'text'):
                execution_text = execution_message.text
            else:
                execution_text = str(execution_message)
            
            execution_data = json.loads(execution_text)
            logger.info(f"Parsed execution data with status: {execution_data.get('status')}")
            return execution_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse execution data JSON: {e}")
            raise ValueError(f"Invalid execution data format: {e}")
        except Exception as e:
            logger.error(f"Error parsing execution data: {e}")
            raise

    def calculate_progress_metrics(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """진행 상황 메트릭 계산"""
        completed_tasks = len(execution_data.get('completed_tasks', []))
        total_tasks = execution_data.get('total_tasks', 0)
        remaining_tasks = execution_data.get('remaining_tasks', 0)
        
        progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        metrics = {
            "completed_tasks": completed_tasks,
            "total_tasks": total_tasks,
            "remaining_tasks": remaining_tasks,
            "progress_percentage": round(progress_percentage, 2),
            "completion_rate": f"{completed_tasks}/{total_tasks}",
            "is_final_task": remaining_tasks <= 1
        }
        
        logger.info(f"Progress metrics: {metrics['completion_rate']} ({metrics['progress_percentage']}%)")
        return metrics

    def analyze_task_quality(self, executed_task: Dict[str, Any]) -> Dict[str, Any]:
        """실행된 작업의 품질 분석"""
        task_status = executed_task.get('status', 'unknown')
        execution_time = executed_task.get('execution_time', 0)
        
        quality_score = 100  # 기본 점수
        issues = []
        recommendations = []
        
        # 상태별 품질 평가
        if task_status == 'failed':
            quality_score = 0
            issues.append("작업 실행 실패")
            recommendations.append("오류 원인 분석 및 재실행 필요")
        elif task_status == 'skipped':
            quality_score = 85
            issues.append("작업 스킵됨 (산출물 이미 존재)")
            recommendations.append("산출물 검증 및 품질 확인 권장")
        elif task_status == 'completed':
            if execution_time > 300:  # 5분 이상
                quality_score -= 15
                issues.append("실행 시간 과다")
                recommendations.append("작업 최적화 또는 분할 고려")
        
        quality_analysis = {
            "quality_score": quality_score,
            "task_status": task_status,
            "execution_time": execution_time,
            "issues": issues,
            "recommendations": recommendations,
            "meets_threshold": quality_score >= self.quality_threshold
        }
        
        return quality_analysis

    async def generate_critique(self, execution_data: Dict[str, Any], 
                              progress_metrics: Dict[str, Any],
                              quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """LLM을 사용하여 실행 결과에 대한 종합적인 비평 생성"""
        try:
            if not hasattr(self, 'critique_prompt') or not self.critique_prompt:
                logger.warning("Critique prompt not provided, using basic evaluation")
                return self._generate_basic_critique(execution_data, progress_metrics, quality_analysis)
                
            if not hasattr(self, 'llm') or self.llm is None:
                logger.warning("LLM not available, using basic evaluation")
                return self._generate_basic_critique(execution_data, progress_metrics, quality_analysis)
                
            critique_prompt = create_prompt_from_component(
                self.critique_prompt,
                default_system_message=self.system_prompt
            )
            
            if critique_prompt is None:
                logger.warning("Failed to create critique prompt, using basic evaluation")
                return self._generate_basic_critique(execution_data, progress_metrics, quality_analysis)
            
            # LLM에게 평가 요청
            chain = critique_prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "execution_data": json.dumps(execution_data, ensure_ascii=False, indent=2),
                "progress_metrics": json.dumps(progress_metrics, ensure_ascii=False, indent=2),
                "quality_analysis": json.dumps(quality_analysis, ensure_ascii=False, indent=2),
                "quality_threshold": self.quality_threshold,
                "suggest_improvements": self.suggest_improvements
            })
            
            logger.info("Generated LLM-based critique")
            return result
            
        except Exception as e:
            logger.error(f"Error generating LLM critique: {e}")
            return self._generate_basic_critique(execution_data, progress_metrics, quality_analysis)

    def _generate_basic_critique(self, execution_data: Dict[str, Any], 
                                progress_metrics: Dict[str, Any],
                                quality_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """기본적인 비평 생성 (LLM 없이)"""
        executed_task = execution_data.get('executed_task', {})
        task_id = executed_task.get('task_id', 'unknown')
        
        # 기본 평가
        overall_rating = "excellent"
        if quality_analysis['quality_score'] < 50:
            overall_rating = "poor"
        elif quality_analysis['quality_score'] < 80:
            overall_rating = "needs_improvement"
        elif quality_analysis['quality_score'] < 95:
            overall_rating = "good"
        
        critique = {
            "task_evaluation": {
                "task_id": task_id,
                "overall_rating": overall_rating,
                "quality_score": quality_analysis['quality_score'],
                "meets_threshold": quality_analysis['meets_threshold'],
                "issues_found": quality_analysis['issues'],
                "recommendations": quality_analysis['recommendations']
            },
            "progress_assessment": {
                "completion_percentage": progress_metrics['progress_percentage'],
                "tasks_remaining": progress_metrics['remaining_tasks'],
                "on_track": progress_metrics['progress_percentage'] > 0,
                "near_completion": progress_metrics['is_final_task']
            },
            "next_actions": [],
            "plan_modifications": {
                "required": not quality_analysis['meets_threshold'],
                "suggestions": []
            }
        }
        
        # 다음 액션 제안
        if progress_metrics['remaining_tasks'] > 0:
            critique["next_actions"].append("계속해서 다음 작업 실행")
        else:
            critique["next_actions"].append("모든 작업 완료 - 최종 검토 필요")
        
        # 계획 수정 제안
        if not quality_analysis['meets_threshold']:
            critique["plan_modifications"]["suggestions"].append(
                f"Task {task_id} 품질 개선 또는 재실행 고려"
            )
        
        return critique

    async def generate_plan_feedback(self, critique_data: Dict[str, Any],
                                   execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """계획 업데이트를 위한 피드백 생성"""
        try:
            if not hasattr(self, 'plan_update_prompt') or not self.plan_update_prompt:
                logger.warning("Plan update prompt not provided, using basic feedback")
                return self._generate_basic_plan_feedback(critique_data, execution_data)
                
            if not hasattr(self, 'llm') or self.llm is None:
                logger.warning("LLM not available, using basic feedback")
                return self._generate_basic_plan_feedback(critique_data, execution_data)
                
            plan_update_prompt = create_prompt_from_component(
                self.plan_update_prompt,
                default_system_message=self.system_prompt
            )
            
            if plan_update_prompt is None:
                logger.warning("Failed to create plan update prompt, using basic feedback")
                return self._generate_basic_plan_feedback(critique_data, execution_data)
            
            # 현재 계획 정보
            current_plan = execution_data.get('critic_data', {}).get('current_plan', {})
            
            # LLM에게 계획 업데이트 요청
            chain = plan_update_prompt | self.llm | JsonOutputParser()
            
            result = await chain.ainvoke({
                "critique_data": json.dumps(critique_data, ensure_ascii=False, indent=2),
                "current_plan": json.dumps(current_plan, ensure_ascii=False, indent=2),
                "execution_data": json.dumps(execution_data, ensure_ascii=False, indent=2)
            })
            
            logger.info("Generated LLM-based plan feedback")
            return result
            
        except Exception as e:
            logger.error(f"Error generating plan feedback: {e}")
            return self._generate_basic_plan_feedback(critique_data, execution_data)

    def _generate_basic_plan_feedback(self, critique_data: Dict[str, Any],
                                    execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """기본적인 계획 피드백 생성"""
        plan_feedback = {
            "update_required": critique_data.get("plan_modifications", {}).get("required", False),
            "feedback_type": "task_completion",
            "completed_task_id": execution_data.get('executed_task', {}).get('task_id'),
            "quality_assessment": critique_data.get("task_evaluation", {}),
            "progress_status": critique_data.get("progress_assessment", {}),
            "recommendations": critique_data.get("next_actions", []),
            "plan_adjustments": critique_data.get("plan_modifications", {}).get("suggestions", [])
        }
        
        return plan_feedback

    async def evaluate_execution(self) -> Message:
        """실행 결과를 평가하고 피드백을 생성"""
        try:
            logger.info("A2A Critic starting evaluation...")
            
            # LLM 초기화
            await self._initialize_llm()
            
            # 실행 데이터 파싱
            execution_data = self.parse_execution_data(self.execution_data)
            
            # 진행 상황 메트릭 계산
            progress_metrics = self.calculate_progress_metrics(execution_data)
            
            # 작업 품질 분석
            executed_task = execution_data.get('executed_task', {})
            quality_analysis = self.analyze_task_quality(executed_task)
            
            # 종합적인 비평 생성
            critique_data = await self.generate_critique(execution_data, progress_metrics, quality_analysis)
            
            # 계획 피드백 생성
            plan_feedback = await self.generate_plan_feedback(critique_data, execution_data)
            
            # 평가 이력에 추가
            evaluation_record = {
                "task_id": executed_task.get('task_id'),
                "evaluation_time": asyncio.get_event_loop().time(),
                "quality_score": quality_analysis['quality_score'],
                "overall_rating": critique_data.get("task_evaluation", {}).get("overall_rating", "unknown")
            }
            self.evaluation_history.append(evaluation_record)
            
            # 최종 결과 구성
            final_result = {
                "critique": critique_data,
                "plan_feedback": plan_feedback,
                "progress_metrics": progress_metrics,
                "quality_analysis": quality_analysis,
                "evaluation_summary": {
                    "task_id": executed_task.get('task_id'),
                    "status": execution_data.get('status'),
                    "quality_score": quality_analysis['quality_score'],
                    "meets_threshold": quality_analysis['meets_threshold'],
                    "progress_percentage": progress_metrics['progress_percentage'],
                    "evaluation_count": len(self.evaluation_history)
                }
            }
            
            result_json = json.dumps(final_result, ensure_ascii=False, indent=2)
            
            return Message(
                text=result_json,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )
            
        except Exception as e:
            error_msg = f"A2A Critic evaluation error: {str(e)}"
            logger.error(error_msg)
            return Message(
                text=error_msg,
                sender=MESSAGE_SENDER_AI,
                sender_name=MESSAGE_SENDER_NAME_AI,
            )

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """모든 필드의 input_types 업데이트"""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
                if key == "agent_llm":
                    build_config[key]["input_types"] = ["LanguageModel"]
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
                if key == "agent_llm":
                    value.input_types = ["LanguageModel"]
        return build_config

    async def update_build_config(
        self, build_config: dotdict, field_value: str, field_name: str | None = None
    ) -> dotdict:
        if field_name in ("agent_llm",):
            build_config["agent_llm"]["value"] = field_value
            provider_info = MODEL_PROVIDERS_DICT.get(field_value)
            if provider_info:
                component_class = provider_info.get("component_class")
                if component_class and hasattr(component_class, "update_build_config"):
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

                for fields in fields_to_delete:
                    self.delete_fields(build_config, fields)

                if field_value == "OpenAI" and not any(field in build_config for field in fields_to_add):
                    build_config.update(fields_to_add)
                else:
                    build_config.update(fields_to_add)
                build_config["agent_llm"]["input_types"] = ["LanguageModel"]
            elif field_value == "Custom":
                self.delete_fields(build_config, ALL_PROVIDER_FIELDS)
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
            
            build_config = self.update_input_types(build_config)

            default_keys = [
                "code", "_type", "agent_llm", "execution_data", "critique_prompt",
                "plan_update_prompt", "quality_threshold", "suggest_improvements",
                "handle_parsing_errors", "verbose", "system_prompt",
            ]
            
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})
