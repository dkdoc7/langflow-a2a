import json
import re
from loguru import logger

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, DropdownInput, IntInput, MessageInput, MessageTextInput, Output, DataInput
from langflow.schema.message import Message
from langflow.schema.data import Data


class ConditionalRouterComponent(Component):
    display_name = "If-Else"
    description = "Routes an input message to a corresponding output based on text comparison."
    documentation: str = "https://docs.langflow.org/components-logic#conditional-router-if-else-component"
    icon = "split"
    name = "ConditionalRouter"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__iteration_updated = False

    inputs = [
        DataInput(
            name="input_value",
            display_name="Loop Item Input",
            info="Loop 컴포넌트의 item 출력을 연결 (JSON 형태의 데이터, status 필드가 포함되어야 함).",
            required=True,
            is_list=False,
        ),
        DropdownInput(
            name="operator",
            display_name="Operator",
            options=[
                "equals",
                "not equals",
                "contains",
                "starts with",
                "ends with",
                "regex",
                "less than",
                "less than or equal",
                "greater than",
                "greater than or equal",
            ],
            info="The operator to apply for comparing the texts.",
            value="equals",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="match_field",
            display_name="Match Field Path",
            info="JSON 필드 경로 (점으로 구분, 예: 'data.0.status' → data[0]['status'])",
            required=True,
        ),
        MessageTextInput(
            name="match_text",
            display_name="Match Text",
            info="The text input to compare against.",
            required=True,
        ),
        BoolInput(
            name="case_sensitive",
            display_name="Case Sensitive",
            info="If true, the comparison will be case sensitive.",
            value=True,
            advanced=True,
        ),

        IntInput(
            name="max_iterations",
            display_name="Max Iterations",
            info="The maximum number of iterations for the conditional router.",
            value=10,
            advanced=True,
        ),
        DropdownInput(
            name="default_route",
            display_name="Default Route",
            options=["true_result", "false_result"],
            info="The default route to take when max iterations are reached.",
            value="false_result",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="True (Message)", name="true_message", method="true_message_response", group_outputs=True),
        Output(display_name="False (Message)", name="false_message", method="false_message_response", group_outputs=True),
        Output(display_name="True (Data)", name="true_result", method="true_response", group_outputs=True),
        Output(display_name="False (Data)", name="false_result", method="false_response", group_outputs=True),
    ]

    def _pre_run_setup(self):
        self.__iteration_updated = False

    def extract_field_from_json(self, input_data) -> str:
        """JSON 입력에서 match_field 경로를 사용하여 값을 추출"""
        try:
            # input_data가 리스트인 경우 첫 번째 요소 처리
            if isinstance(input_data, list) and len(input_data) > 0:
                input_data = input_data[0]
            
            # JSON 데이터 추출
            json_data = None
            
            # DataInput으로 받은 Data 객체 처리
            if hasattr(input_data, 'data'):
                if isinstance(input_data.data, dict):
                    json_data = input_data.data
                elif isinstance(input_data.data, str):
                    json_data = json.loads(input_data.data)
            elif hasattr(input_data, 'text'):
                json_data = json.loads(input_data.text)
            elif hasattr(input_data, 'content'):
                json_data = json.loads(input_data.content)
            elif hasattr(input_data, 'item'):
                json_data = json.loads(input_data.item)
            else:
                # 문자열인 경우 직접 JSON 파싱
                json_data = json.loads(str(input_data))
            
            if json_data is None:
                return 'unknown'
            
            # match_field 경로를 사용하여 값 추출
            return self._get_nested_value(json_data, self.match_field)
                
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return 'unknown'
    
    def _get_nested_value(self, data: dict, field_path: str) -> str:
        """
        점(.)으로 구분된 경로를 사용하여 중첩된 JSON 구조에서 값을 추출
        예: "data.0.status" -> data["data"][0]["status"]
        """
        try:
            current = data
            path_parts = field_path.split('.')
            
            for part in path_parts:
                # 숫자인 경우 리스트 인덱스로 처리
                if part.isdigit():
                    current = current[int(part)]
                else:
                    current = current[part]
            
            return str(current) if current is not None else 'unknown'
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"필드 경로 '{field_path}' 접근 오류: {e}")
            return 'unknown'

    def evaluate_condition(self, input_value: str, match_text: str, operator: str, *, case_sensitive: bool) -> bool:
        if not case_sensitive and operator != "regex":
            input_value = input_value.lower()
            match_text = match_text.lower()

        if operator == "equals":
            return input_value == match_text
        if operator == "not equals":
            return input_value != match_text
        if operator == "contains":
            return match_text in input_value
        if operator == "starts with":
            return input_value.startswith(match_text)
        if operator == "ends with":
            return input_value.endswith(match_text)
        if operator == "regex":
            try:
                return bool(re.match(match_text, input_value))
            except re.error:
                return False  # Return False if the regex is invalid
        if operator in ["less than", "less than or equal", "greater than", "greater than or equal"]:
            try:
                input_num = float(input_value)
                match_num = float(match_text)
                if operator == "less than":
                    return input_num < match_num
                if operator == "less than or equal":
                    return input_num <= match_num
                if operator == "greater than":
                    return input_num > match_num
                if operator == "greater than or equal":
                    return input_num >= match_num
            except ValueError:
                return False  # Invalid number format for comparison
        return False

    def iterate_and_stop_once(self, route_to_stop: str):
        if not self.__iteration_updated:
            self.update_ctx({f"{self._id}_iteration": self.ctx.get(f"{self._id}_iteration", 0) + 1})
            self.__iteration_updated = True
            if self.ctx.get(f"{self._id}_iteration", 0) >= self.max_iterations and route_to_stop == self.default_route:
                route_to_stop = "true_result" if route_to_stop == "false_result" else "false_result"
            self.stop(route_to_stop)

    def true_response(self) -> Data:
        # JSON에서 match_field 경로의 값 추출
        field_value = self.extract_field_from_json(self.input_value)
        result = self.evaluate_condition(
            field_value, self.match_text, self.operator, case_sensitive=self.case_sensitive
        )
        if result:
            self.iterate_and_stop_once("false_result")
            return self.input_value
        self.iterate_and_stop_once("true_result")
        return Data(value="")


    def false_response(self) -> Data:
        # JSON에서 match_field 경로의 값 추출
        field_value = self.extract_field_from_json(self.input_value)
        
        result = self.evaluate_condition(
            field_value, self.match_text, self.operator, case_sensitive=self.case_sensitive
        )
        if result:
            self.iterate_and_stop_once("true_result")
            return self.input_value
        self.iterate_and_stop_once("false_result")
        return Data(value="")

    def true_message_response(self) -> Message:
        # JSON에서 match_field 경로의 값 추출
        field_value = self.extract_field_from_json(self.input_value)
        result = self.evaluate_condition(
            field_value, self.match_text, self.operator, case_sensitive=self.case_sensitive
        )
        if result:
            self.iterate_and_stop_once("false_result")
            # Data 객체를 Message로 변환
            if hasattr(self.input_value, 'data'):
                return Message(text=json.dumps(self.input_value.data, ensure_ascii=False, indent=2))
            elif hasattr(self.input_value, 'text'):
                return Message(text=self.input_value.text)
            else:
                return Message(text=str(self.input_value))
        self.iterate_and_stop_once("true_result")
        return Message(text="")

    def false_message_response(self) -> Message:
        # JSON에서 match_field 경로의 값 추출
        field_value = self.extract_field_from_json(self.input_value)
        
        result = self.evaluate_condition(
            field_value, self.match_text, self.operator, case_sensitive=self.case_sensitive
        )
        if result:
            self.iterate_and_stop_once("true_result")
            # Data 객체를 Message로 변환
            if hasattr(self.input_value, 'data'):
                return Message(text=json.dumps(self.input_value.data, ensure_ascii=False, indent=2))
            elif hasattr(self.input_value, 'text'):
                return Message(text=self.input_value.text)
            else:
                return Message(text=str(self.input_value))
        self.iterate_and_stop_once("false_result")
        return Message(text="")

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None) -> dict:
        if field_name == "operator":
            if field_value == "regex":
                build_config.pop("case_sensitive", None)
            elif "case_sensitive" not in build_config:
                case_sensitive_input = next(
                    (input_field for input_field in self.inputs if input_field.name == "case_sensitive"), None
                )
                if case_sensitive_input:
                    build_config["case_sensitive"] = case_sensitive_input.to_dict()
        return build_config

