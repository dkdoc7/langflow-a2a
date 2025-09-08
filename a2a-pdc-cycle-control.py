from langflow.custom.custom_component.component import Component
from langflow.inputs.inputs import HandleInput, IntInput, BoolInput, MultilineInput, StrInput, MessageInput
from langflow.schema.data import Data
from langflow.schema.message import Message
from langflow.template.field.base import Output
import json
from typing import Dict, List, Any, Optional


class PDCCycleController(Component):
    display_name = "PDC Cycle Controller"
    description = (
        "Plan-Do-Critic 사이클을 제어하는 컴포넌트. 작업 계획을 받아서 각 작업의 상태를 확인하고 "
        "순차적으로 실행하며, 출력 검증을 통해 중복 작업을 방지합니다."
    )
    icon = "refresh-cw"

    inputs = [
        MessageInput(
            name="plan_feedback",
            display_name="작업 피드백",
            info="MessageHandler를 통해 입력받은 작업 계획 (JSON 형태)",
        ),

        HandleInput(
            name="llm_validator",
            display_name="LLM Model",
            info="출력 검증을 위한 LLM Model",
            input_types=["LanguageModel"],
            required=False,
        ),

        IntInput(
            name="max_iterations",
            display_name="최대 반복 횟수",
            info="최대 사이클 반복 횟수",
            value=10,
        ),

        MultilineInput(
            name="validation_prompt_template",
            display_name="검증 프롬프트 템플릿",
            info="출력 검증을 위한 LLM 프롬프트 템플릿",
            value="""다음 작업의 예상 출력이 이미 기존 출력 목록에 포함되어 있는지 판단해주세요:

작업 정보:
- 작업 ID: {task_id}
- 작업 설명: {task_description}
- 예상 출력: {expected_output}

기존 출력 목록:
{existing_outputs}

이 작업의 출력이 이미 기존 출력에 포함되어 있다면 "YES"를, 그렇지 않다면 "NO"를 응답해주세요.
응답은 YES 또는 NO만 작성해주세요.""",
        ),
    ]

    outputs = [
        Output(
            display_name="현재 작업 (Data)",
            name="current_task",
            method="get_current_task",
            allows_loop=True,
        ),
        Output(
            display_name="현재 작업 (Message)",
            name="current_task_message",
            method="get_current_task_message",
            allows_loop=True,
        ),
        Output(
            display_name="작업 계획 (Text)",
            name="plan_text",
            method="get_plan_text",
            allows_loop=True,
        ),
        Output(
            display_name="완료 상태",
            name="done",
            method="check_completion",
        ),
    ]

    def initialize_controller(self) -> None:
        """컨트롤러 초기화"""
        if self.ctx.get(f"{self._id}_initialized", False):
            return

        # 초기 상태 설정
        self.update_ctx({
            f"{self._id}_current_iteration": 0,
            f"{self._id}_plan": {
                "goal": "",
                "assumptions": [],
                "updated": False,
                "work_breakdown": []
            },
            f"{self._id}_task_outputs_registry": {},
            f"{self._id}_cycle_completed": False,
            f"{self._id}_loop_active": True,
            f"{self._id}_waiting_for_feedback": False,
            f"{self._id}_initialized": True,
        })

    def parse_plan_feedback(self) -> Dict[str, Any]:
        """비평 결과에서 updated_plan을 추출하여 작업 계획 파싱"""
        try:
            if isinstance(self.plan_feedback, Message):
                input_text = self.plan_feedback.text
            else:
                input_text = str(self.plan_feedback)
            
            # JSON 형태로 파싱 시도 (비평 결과)
            if input_text.strip().startswith('{'):
                critique_data = json.loads(input_text)
                
                # updated_plan 필드가 있는지 확인
                if "updated_plan" in critique_data:
                    plan_data = critique_data["updated_plan"]
                    self.log("비평 결과에서 업데이트된 작업 계획을 추출했습니다.")
                else:
                    # updated_plan이 없는 경우, 전체를 작업 계획으로 간주
                    plan_data = critique_data
            else:
                # 텍스트 형태인 경우 기본 형태로 변환
                plan_data = {
                    "goal": input_text,
                    "assumptions": [],
                    "updated": False,
                    "work_breakdown": [{
                        "id": "T1",
                        "title": input_text,
                        "agent": "default_agent",
                        "skills": [],
                        "inputs": [],
                        "outputs": [],
                        "dod": [],
                        "est_hours": 1.0,
                        "dependencies": [],
                        "parallelizable": True,
                        "risk": [],
                        "mitigation": [],
                        "bus": {
                            "use_bus": False,
                            "topics": [],
                            "role": "consumer",
                            "message_schema": {},
                            "qos": {},
                            "contracts": []
                        },
                        "status": "pending",
                        "actual_output": None
                    }]
                }
            
            # 작업 계획 스키마 검증 및 기본값 설정
            if "work_breakdown" not in plan_data:
                plan_data["work_breakdown"] = []
            
            if "goal" not in plan_data:
                plan_data["goal"] = "작업 수행"
            
            if "assumptions" not in plan_data:
                plan_data["assumptions"] = []
            
            if "updated" not in plan_data:
                plan_data["updated"] = True  # 비평 결과에서 온 것이므로 업데이트된 것으로 표시
            
            # 각 작업에 필수 필드 확인 및 기본값 설정
            for task in plan_data["work_breakdown"]:
                self._ensure_task_fields(task)
            
            return plan_data
        except (json.JSONDecodeError, Exception) as e:
            self.log(f"작업 계획 파싱 오류: {e}")
            return {
                "goal": "기본 작업",
                "assumptions": [],
                "updated": False,
                "work_breakdown": []
            }
    
    def _ensure_task_fields(self, task: Dict[str, Any]) -> None:
        """작업 객체의 필수 필드 확인 및 기본값 설정"""
        required_fields = {
            "id": f"T{len(self.ctx.get(f'{self._id}_tasks', {}).get('work_breakdown', [])) + 1}",
            "title": "제목 없음",
            "agent": "default_agent",
            "skills": [],
            "inputs": [],
            "outputs": [],
            "dod": [],
            "est_hours": 1.0,
            "dependencies": [],
            "parallelizable": True,
            "risk": [],
            "mitigation": [],
            "bus": {
                "use_bus": False,
                "topics": [],
                "role": "consumer",
                "message_schema": {},
                "qos": {},
                "contracts": []
            },
            "status": "pending",
            "actual_output": None
        }
        
        for field, default_value in required_fields.items():
            if field not in task:
                task[field] = default_value

    def should_continue_loop(self) -> bool:
        """루프를 계속해야 하는지 판단"""
        # 사이클이 완료된 경우 루프 중단
        if self.ctx.get(f"{self._id}_cycle_completed", False):
            return False
        
        # 최대 반복 횟수 도달한 경우 루프 중단
        current_iteration = self.ctx.get(f"{self._id}_current_iteration", 0)
        if current_iteration >= self.max_iterations:
            return False
        
        # 모든 작업이 완료된 경우 루프 중단
        if self.all_tasks_completed():
            return False
        
        return True

    def stop_loop(self, output_name: str) -> None:
        """특정 출력 포트에 대해 루프 중단"""
        self.stop(output_name)
        if output_name in ["current_task", "current_task_message"]:
            # 주요 출력들이 중단되면 루프 비활성화
            self.update_ctx({f"{self._id}_loop_active": False})

    def start_loop(self, output_name: str) -> None:
        """특정 출력 포트에 대해 루프 시작"""
        self.start(output_name)

    def update_dependency_for_loop(self) -> None:
        """루프를 위한 종속성 업데이트"""
        # plan_feedback 입력과의 종속성 설정
        feedback_dependency_id = self.get_incoming_edge_by_target_param("plan_feedback")
        if feedback_dependency_id:
            for output_name in ["current_task", "current_task_message", "plan_text"]:
                if feedback_dependency_id not in self.graph.run_manager.run_predecessors.get(self._id, []):
                    if self._id not in self.graph.run_manager.run_predecessors:
                        self.graph.run_manager.run_predecessors[self._id] = []
                    self.graph.run_manager.run_predecessors[self._id].append(feedback_dependency_id)

    def update_task_plan(self, new_plan: Dict[str, Any]) -> None:
        """작업 계획 업데이트"""
        existing_plan = self.ctx.get(f"{self._id}_plan", {})
        
        # 기존 계획과 새 계획을 비교하여 업데이트
        if existing_plan != new_plan:
            # 기존 작업들의 상태 정보 유지
            if existing_plan.get("work_breakdown"):
                existing_tasks = {task["id"]: task for task in existing_plan["work_breakdown"]}
                for new_task in new_plan["work_breakdown"]:
                    task_id = new_task["id"]
                    if task_id in existing_tasks:
                        # 기존 작업의 상태와 실제 출력 유지
                        new_task["status"] = existing_tasks[task_id].get("status", "pending")
                        new_task["actual_output"] = existing_tasks[task_id].get("actual_output")
            
            self.log("작업 계획이 업데이트되었습니다.")
            self.update_ctx({f"{self._id}_plan": new_plan})

    def get_next_available_task(self) -> Optional[Dict[str, Any]]:
        """다음 실행 가능한 작업을 찾기"""
        plan = self.ctx.get(f"{self._id}_plan", {})
        tasks = plan.get("work_breakdown", [])
        
        for task in tasks:
            if task["status"] == "pending":
                # 종속성 확인
                dependencies_met = True
                for dep_id in task.get("dependencies", []):
                    dep_task = next((t for t in tasks if t["id"] == dep_id), None)
                    if not dep_task or dep_task["status"] != "done":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    return task
        
        return None

    def validate_output_exists(self, task: Dict[str, Any]) -> bool:
        """LLM을 이용하여 작업 출력이 이미 존재하는지 검증 (항상 활성화)"""
        try:
            # 기존 출력 목록 가져오기
            existing_outputs = self.ctx.get(f"{self._id}_task_outputs_registry", {})
            
            if not existing_outputs:
                return False
            
            # LLM 검증기가 연결되어 있는 경우 실제 LLM 호출
            if hasattr(self, 'llm_validator') and self.llm_validator:
                # 검증 프롬프트 생성 (새로운 스키마에 맞게)
                prompt = self.validation_prompt_template.format(
                    task_id=task.get("id", ""),
                    task_description=task.get("title", ""),
                    expected_output=", ".join(task.get("outputs", [])),
                    existing_outputs=json.dumps(existing_outputs, ensure_ascii=False, indent=2)
                )
                
                try:
                    # LanguageModel 객체를 통한 LLM 호출
                    if hasattr(self.llm_validator, 'invoke'):
                        # LanguageModel의 invoke 메서드 사용
                        llm_response = self.llm_validator.invoke(prompt)
                        if hasattr(llm_response, 'content'):
                            response_text = llm_response.content.strip().upper()
                        else:
                            response_text = str(llm_response).strip().upper()
                    elif hasattr(self.llm_validator, '__call__'):
                        # callable 객체인 경우
                        llm_response = self.llm_validator(prompt)
                        response_text = str(llm_response).strip().upper()
                    else:
                        # 기타 경우
                        response_text = str(self.llm_validator).strip().upper()
                    
                    return "YES" in response_text
                except Exception as llm_error:
                    self.log(f"LLM 검증 호출 오류: {llm_error}")
            
            # LLM이 없는 경우 간단한 키워드 매칭으로 대체
            task_title = task.get("title", "").lower()
            task_outputs = [output.lower() for output in task.get("outputs", [])]
            
            for output_key, output_value in existing_outputs.items():
                output_str = str(output_value).lower()
                if task_title in output_str:
                    return True
                for expected_output in task_outputs:
                    if expected_output in output_str:
                        return True
            
            return False
        except Exception as e:
            self.log(f"출력 검증 오류: {e}")
            return False



    def mark_task_completed(self, task: Dict[str, Any], output: Any) -> None:
        """작업을 완료로 표시하고 출력 저장"""
        plan = self.ctx.get(f"{self._id}_plan", {})
        tasks = plan.get("work_breakdown", [])
        task_outputs = self.ctx.get(f"{self._id}_task_outputs_registry", {})
        
        # 작업 상태 업데이트
        for t in tasks:
            if t["id"] == task["id"]:
                t["status"] = "done"
                t["actual_output"] = output
                break
        
        # 출력 저장소에 등록
        task_outputs[task["id"]] = output
        
        plan["work_breakdown"] = tasks
        self.update_ctx({
            f"{self._id}_plan": plan,
            f"{self._id}_task_outputs_registry": task_outputs,
        })

    def all_tasks_completed(self) -> bool:
        """모든 작업이 완료되었는지 확인"""
        plan = self.ctx.get(f"{self._id}_plan", {})
        tasks = plan.get("work_breakdown", [])
        return all(task["status"] == "done" for task in tasks) if tasks else False

    def increment_iteration(self) -> bool:
        """반복 횟수 증가 및 최대 반복 횟수 확인 (루프 제어에서는 사용하지 않음)"""
        current_iteration = self.ctx.get(f"{self._id}_current_iteration", 0)
        # 주의: 이 메서드는 get_current_task에서 이미 처리되므로 여기서는 단순 확인만
        return current_iteration >= self.max_iterations

    def get_current_task(self) -> Data:
        """현재 실행할 작업 반환 (루프 지원)"""
        self.initialize_controller()
        
        # 루프 지속 여부 확인
        if not self.should_continue_loop():
            self.stop_loop("current_task")
            self.start_loop("done")
            return Data(text=json.dumps({
                "type": "cycle_completed",
                "message": "모든 작업이 완료되었습니다.",
                "final_status": self.get_final_status()
            }, ensure_ascii=False))
        
        # 작업 계획 파싱 및 업데이트 (plan_feedback에서)
        new_plan = self.parse_plan_feedback()
        if new_plan:
            self.update_task_plan(new_plan)
            # 피드백을 받았으므로 다시 루프 실행
            self.update_ctx({f"{self._id}_waiting_for_feedback": False})
        
        # 피드백 대기 중이면 루프 일시 중단
        if self.ctx.get(f"{self._id}_waiting_for_feedback", False):
            return Data(text=json.dumps({
                "type": "waiting_feedback",
                "message": "작업 피드백을 대기 중입니다.",
                "current_status": self.get_current_status()
            }, ensure_ascii=False))
        
        # 최대 반복 횟수 확인
        current_iteration = self.ctx.get(f"{self._id}_current_iteration", 0)
        if current_iteration >= self.max_iterations:
            self.update_ctx({f"{self._id}_cycle_completed": True})
            self.stop_loop("current_task")
            self.start_loop("done")
            return Data(text=json.dumps({
                "type": "cycle_terminated",
                "message": "최대 반복 횟수에 도달했습니다. 사이클을 강제 종료합니다.",
                "iteration": current_iteration,
                "max_iterations": self.max_iterations,
                "final_status": self.get_final_status()
            }, ensure_ascii=False))
        
        # 모든 작업 완료 확인
        if self.all_tasks_completed():
            self.update_ctx({f"{self._id}_cycle_completed": True})
            self.stop_loop("current_task")
            self.start_loop("done")
            return Data(text=json.dumps({
                "type": "all_tasks_completed",
                "message": "모든 작업이 완료되었습니다.",
                "final_status": self.get_final_status()
            }, ensure_ascii=False))
        
        # 다음 실행 가능한 작업 찾기
        next_task = self.get_next_available_task()
        if not next_task:
            # 실행 가능한 작업이 없으면 잠시 대기
            return Data(text=json.dumps({
                "type": "no_available_task",
                "message": "실행 가능한 작업이 없습니다. 종속성을 확인하거나 피드백을 대기합니다.",
                "current_status": self.get_current_status()
            }, ensure_ascii=False))
        
        # 출력 검증
        if self.validate_output_exists(next_task):
            output_message = f"작업 '{next_task['id']}'의 출력이 이미 존재합니다. 다음 작업으로 이동합니다."
            self.log(output_message)
            # 작업을 완료로 표시
            self.mark_task_completed(next_task, "이미 완료된 작업")
            
            # 재귀적으로 다음 작업 찾기 (루프 내에서)
            return self.get_current_task()
        
        # 현재 작업을 진행 중으로 표시
        plan = self.ctx.get(f"{self._id}_plan", {})
        tasks = plan.get("work_breakdown", [])
        for task in tasks:
            if task["id"] == next_task["id"]:
                task["status"] = "in_progress"
                break
        plan["work_breakdown"] = tasks
        self.update_ctx({
            f"{self._id}_plan": plan,
            f"{self._id}_waiting_for_feedback": True,  # 작업 배정 후 피드백 대기
            f"{self._id}_current_iteration": current_iteration + 1
        })
        
        # 루프 종속성 업데이트
        self.update_dependency_for_loop()
        
        # 현재 작업 반환 (작업 위임 에이전트로 전달)
        task_message = {
            "type": "task_assignment",
            "task": next_task,
            "iteration": current_iteration + 1,
            "context": {
                "total_tasks": len(tasks),
                "completed_tasks": len([t for t in tasks if t["status"] == "done"]),
                "task_outputs_registry": self.ctx.get(f"{self._id}_task_outputs_registry", {})
            }
        }
        
        return Data(text=json.dumps(task_message, ensure_ascii=False))

    def get_current_task_message(self) -> Message:
        """현재 실행할 작업을 Message 형태로 반환 (AgentComponent 연결용)"""
        # 루프 지속 여부 확인
        if not self.should_continue_loop():
            self.stop_loop("current_task_message")
            return Message(text="모든 작업이 완료되었습니다.")
        
        # 현재 작업 정보를 Data로 가져오기
        current_task_data = self.get_current_task()
        
        try:
            # JSON 파싱 시도
            task_info = json.loads(current_task_data.text)
            
            if task_info.get("type") == "task_assignment":
                # 작업 할당인 경우 AgentComponent에 적합한 메시지 형태로 변환
                task = task_info["task"]
                
                # 에이전트를 위한 구조화된 메시지 생성
                agent_message = f"""작업 할당: {task.get('title', '제목 없음')}

## 작업 상세 정보
- **작업 ID**: {task.get('id', '')}
- **담당 에이전트**: {task.get('agent', 'default_agent')}
- **예상 소요 시간**: {task.get('est_hours', 0)}시간

## 필요 스킬
{chr(10).join(f"- {skill}" for skill in task.get('skills', [])) if task.get('skills') else "- 특별한 스킬 요구사항 없음"}

## 입력 요구사항
{chr(10).join(f"- {inp}" for inp in task.get('inputs', [])) if task.get('inputs') else "- 특별한 입력 요구사항 없음"}

## 기대 출력
{chr(10).join(f"- {out}" for out in task.get('outputs', [])) if task.get('outputs') else "- 출력 명세 없음"}

## 완료 기준 (DoD)
{chr(10).join(f"- {dod}" for dod in task.get('dod', [])) if task.get('dod') else "- 완료 기준 명세 없음"}

## 리스크 및 완화 방안
### 식별된 리스크:
{chr(10).join(f"- {risk}" for risk in task.get('risk', [])) if task.get('risk') else "- 식별된 리스크 없음"}

### 완화 방안:
{chr(10).join(f"- {mitigation}" for mitigation in task.get('mitigation', [])) if task.get('mitigation') else "- 완화 방안 없음"}

## 병렬 실행 가능 여부
{task.get('parallelizable', True) and "병렬 실행 가능" or "순차 실행 필요"}

## 컨텍스트 정보
- **현재 반복**: {task_info.get('iteration', 0)}
- **전체 작업 수**: {task_info.get('context', {}).get('total_tasks', 0)}
- **완료된 작업 수**: {task_info.get('context', {}).get('completed_tasks', 0)}

## 작업 지시
위 정보를 바탕으로 할당된 작업을 수행해주세요. 작업 완료 시에는 다음 JSON 형태로 결과를 보고해주세요:

```json
{{
    "task_id": "{task.get('id', '')}",
    "status": "done",
    "output": "작업 결과 상세 설명",
    "issues": "발생한 문제점 (있는 경우)",
    "recommendations": "향후 개선 사항 (있는 경우)"
}}
```
"""
                
                return Message(text=agent_message)
            
            else:
                # 사이클 완료, 종료, 오류 등의 경우
                return Message(text=task_info.get("message", current_task_data.text))
                
        except (json.JSONDecodeError, Exception):
            # JSON 파싱 실패시 원본 텍스트 사용
            return Message(text=current_task_data.text)

    def get_plan_text(self) -> str:
        """작업 계획을 텍스트 형태로 출력 (TextOutput 연결용)"""
        # 루프 지속 여부와 관계없이 현재 계획 상태 출력
        plan = self.ctx.get(f"{self._id}_plan", {})
        
        if not plan or not plan.get("work_breakdown"):
            return "작업 계획이 설정되지 않았습니다."
        
        # 작업 계획을 읽기 쉬운 텍스트 형태로 포맷팅
        plan_text = f"""
=== 작업 계획 ===

🎯 목표: {plan.get('goal', '목표 없음')}

📋 가정사항:
{chr(10).join(f"• {assumption}" for assumption in plan.get('assumptions', [])) if plan.get('assumptions') else "• 특별한 가정사항 없음"}

🔄 계획 업데이트 여부: {'✓ 업데이트됨' if plan.get('updated', False) else '✗ 초기 계획'}

📊 작업 현황:
• 총 작업 수: {len(plan['work_breakdown'])}
• 완료된 작업: {len([t for t in plan['work_breakdown'] if t['status'] == 'done'])}
• 진행 중인 작업: {len([t for t in plan['work_breakdown'] if t['status'] == 'in_progress'])}
• 대기 중인 작업: {len([t for t in plan['work_breakdown'] if t['status'] == 'pending'])}

📝 작업 상세:
"""
        
        for i, task in enumerate(plan['work_breakdown'], 1):
            status_icon = {
                'pending': '⏳',
                'in_progress': '🔄',
                'done': '✅',
                'failed': '❌'
            }.get(task.get('status', 'pending'), '❓')
            
            plan_text += f"""
{i}. {status_icon} {task.get('title', '제목 없음')} ({task.get('id', '')})
   담당: {task.get('agent', 'default_agent')}
   예상 소요시간: {task.get('est_hours', 0)}시간
   병렬 실행: {'가능' if task.get('parallelizable', True) else '불가능'}
   
   필요 스킬: {', '.join(task.get('skills', [])) if task.get('skills') else '없음'}
   
   입력 요구사항:
   {chr(10).join(f"   • {inp}" for inp in task.get('inputs', [])) if task.get('inputs') else '   • 없음'}
   
   기대 출력:
   {chr(10).join(f"   • {out}" for out in task.get('outputs', [])) if task.get('outputs') else '   • 없음'}
   
   완료 기준:
   {chr(10).join(f"   • {dod}" for dod in task.get('dod', [])) if task.get('dod') else '   • 없음'}
   
   종속성: {', '.join(task.get('dependencies', [])) if task.get('dependencies') else '없음'}
   
   리스크: {', '.join(task.get('risk', [])) if task.get('risk') else '없음'}
   
   완화 방안: {', '.join(task.get('mitigation', [])) if task.get('mitigation') else '없음'}
"""
            
            if task.get('actual_output'):
                plan_text += f"   실제 출력: {task['actual_output']}\n"
        
        # 현재 사이클 정보 추가
        current_iteration = self.ctx.get(f"{self._id}_current_iteration", 0)
        cycle_completed = self.ctx.get(f"{self._id}_cycle_completed", False)
        
        plan_text += f"""

=== 사이클 정보 ===
현재 반복: {current_iteration}/{self.max_iterations}
사이클 상태: {'완료' if cycle_completed else '진행 중'}

업데이트 시간: {self._get_current_time()}
"""
        
        return plan_text.strip()
    
    def _get_current_time(self) -> str:
        """현재 시간을 문자열로 반환"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_current_status(self) -> Dict[str, Any]:
        """현재 상태 정보 반환"""
        plan = self.ctx.get(f"{self._id}_plan", {})
        tasks = plan.get("work_breakdown", [])
        return {
            "current_iteration": self.ctx.get(f"{self._id}_current_iteration", 0),
            "goal": plan.get("goal", ""),
            "assumptions": plan.get("assumptions", []),
            "updated": plan.get("updated", False),
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t["status"] == "done"]),
            "in_progress_tasks": len([t for t in tasks if t["status"] == "in_progress"]),
            "pending_tasks": len([t for t in tasks if t["status"] == "pending"]),
            "tasks": tasks
        }
    
    def get_final_status(self) -> Dict[str, Any]:
        """최종 상태 정보 반환"""
        plan = self.ctx.get(f"{self._id}_plan", {})
        tasks = plan.get("work_breakdown", [])
        task_outputs = self.ctx.get(f"{self._id}_task_outputs_registry", {})
        
        return {
            "goal": plan.get("goal", ""),
            "assumptions": plan.get("assumptions", []),
            "total_iterations": self.ctx.get(f"{self._id}_current_iteration", 0),
            "total_tasks": len(tasks),
            "completed_tasks": len([t for t in tasks if t["status"] == "done"]),
            "failed_tasks": len([t for t in tasks if t["status"] == "failed"]),
            "tasks_summary": [
                {
                    "id": task["id"],
                    "title": task.get("title", ""),
                    "agent": task.get("agent", ""),
                    "status": task["status"],
                    "estimated_hours": task.get("est_hours", 0),
                    "outputs": task.get("outputs", []),
                    "actual_output": task.get("actual_output"),
                    "has_output": task["id"] in task_outputs,
                    "dod_met": bool(task.get("actual_output")) and task["status"] == "done"
                }
                for task in tasks
            ],
            "outputs_count": len(task_outputs)
        }

    def check_completion(self) -> Data:
        """완료 상태 확인"""
        # 루프가 완료되어야 이 출력이 활성화됨
        if not self.should_continue_loop():
            self.start_loop("done")
            return Data(text=json.dumps({
                "type": "cycle_completed",
                "message": "PDC 사이클이 완료되었습니다.",
                "final_status": self.get_final_status()
            }, ensure_ascii=False))
        else:
            self.stop_loop("done")
            return Data(text=json.dumps({
                "type": "cycle_in_progress",
                "message": "PDC 사이클이 진행 중입니다.",
                "current_status": self.get_current_status()
            }, ensure_ascii=False))

