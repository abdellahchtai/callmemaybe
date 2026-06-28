from llm_sdk.llm_sdk import Small_LLM_Model
from .parsing import FunctionDef, ValidPrompt
from .functions_name import FunctionName
from .functions_parameters import FunctionParam
import numpy as np
import os
import json


class Manager:

    def __init__(self, model_name: str):
        self.llm = Small_LLM_Model(model_name=model_name)

    def main(self, prompts: list[ValidPrompt], avail_fts: list[FunctionDef]):

        function_map = {
            function_def.name: {
                "params": function_def.parameters,
                "descrp": function_def.description
                }
            for function_def in avail_fts}

        data = []

        encoded_functions = [self.encode(elm) for elm in function_map.keys()]

        find_name = FunctionName(encoded_functions, avail_fts, self)
        find_param = FunctionParam(self, function_map)
        valid_int, valid_bool = find_param.encoded_valid_token()

        for prompt in prompts:

            fn = find_name.get_fn(prompt.prompt)
            data.append(find_param.get_parameters(
                fn, prompt.prompt, valid_int, valid_bool))

        self.output(data)

    def encode(self, to_encode: str) -> list[int]:

        tokens = self.llm.encode(to_encode).squeeze().tolist()

        return tokens if isinstance(tokens, list) else [tokens]

    def constraint_decode(self, logits: list[float],
                          to_check: list[int]) -> int:

        if to_check is False:
            return np.argmax(logits)

        return max(to_check, key=logits.__getitem__)

    def output(self, data: dict):

        os.makedirs('data/output', exist_ok=True)

        with open("data/output/output.json", "w") as f:
            json.dump(data, f, indent=2)
