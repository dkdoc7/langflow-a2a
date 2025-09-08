from langflow.base.io.text import TextComponent
from langflow.io import MultilineInput, Output
from langflow.schema.message import Message
from langflow.schema.dataframe import DataFrame

import json
import pandas as pd

class TextOutputComponent(TextComponent):
    display_name = "JSON Parser"
    description = "Parse text to JOSN"
    icon = "type"
    name = "JsonParser"

    inputs = [
        MultilineInput(
            name="input_value",
            display_name="Inputs",
            info="Text to be parse as JSON.",
        ),
    ]
    outputs = [
        Output(display_name="Output JSON", name="json", method="json_response"),
    ]

    def json_response(self) -> DataFrame:
        jsonStr = json.loads(self.input_value)
        df = DataFrame(jsonStr)
        self.status = self.input_value
        return df
