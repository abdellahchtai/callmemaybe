from llm_sdk.llm_sdk import Small_LLM_Model
from .parsing import FunctionDef, ValidPrompt, TypeLiteral
import numpy as np
import os
import json
from typing import Optional, Any, Union


class Manager:

    def __init__(self, model_name: str):
        self.llm = Small_LLM_Model(model_name=model_name)

    def main(self, prompts: list[ValidPrompt],
             avail_fts: list[FunctionDef]) -> list[dict]:

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

        for i, prompt in enumerate(prompts, 1):

            print()
            print('---' * 70)
            print(f"Processing prompt {i}/{len(prompts)}: {prompt.prompt}.\n")
            print(" Step 1: Find function name.")
            fn = find_name.get_fn(prompt.prompt)
            print("   Function name found successfully.\n")
            print(" Step 2: Find parameters.")
            data.append(find_param.get_parameters(
                fn, prompt.prompt, valid_int, valid_bool))
            print("  Parameters function found successfully.")

        return data

    def encode(self, to_encode: str) -> list[int]:

        tokens = self.llm.encode(to_encode).squeeze().tolist()

        return tokens if isinstance(tokens, list) else [tokens]

    def constraint_decode(self, logits: list[float],
                          to_check: Optional[list[int]]) -> int:

        if to_check is None:
            return int(np.argmax(logits))

        return max(to_check, key=logits.__getitem__)

    def output_file(self, data: list[dict], path: str) -> None:

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n\n🎉🎊Congrats, To see the expected output go to {path}.")


class FunctionName:

    def __init__(self, encoded_fn: list[list[int]],
                 avail_fts: list[FunctionDef], manager: Manager) -> None:

        self.encoded_fn = encoded_fn
        self.manager = manager
        self.encoded_prompt = self.prompt(str(avail_fts))

    def get_fn(self, user_prompt: str) -> list[int]:

        full_prompt = self.encoded_prompt + self.user_prompt(user_prompt)
        return self.get_fn_ids(full_prompt)

    def prompt(self, function_definitions: str) -> list[int]:

        main_prompt = (
            f"Here are the available functions:{function_definitions}"
            "\nInstructions:\n- Rea"
            "d the user's prompt carefully.\n- Match the user's intent ONLY ag"
            "ainst the function descriptions above.\n- You MUST return the fun"
            "ction name EXACTLY as written above, nothing else.\n- Do NOT expl"
            "ain, do NOT add punctuation, do NOT guess.\nUser's prompt: ")

        return self.manager.encode(main_prompt)

    def user_prompt(self, user_prompt: str) -> list[int]:

        main_prompt = (
            f"{user_prompt}\nBased strictly on the function descriptions above"
            ", the single most appropriate function to call for the user "
            "prompt is: ")

        return self.manager.encode(main_prompt)

    def get_fn_ids(self, encoded_prompt: list[int]) -> list[int]:

        answer_start = len(encoded_prompt)
        i = 0
        encoded_functions = self.encoded_fn

        while encoded_functions:

            if i >= len(encoded_functions[0]):
                break

            to_keep = [elm[i] for elm in encoded_functions]
            logits = self.manager.llm.get_logits_from_input_ids(encoded_prompt)
            token = self.manager.constraint_decode(
                logits, to_keep)

            encoded_functions = [elm for elm in encoded_functions
                                 if elm[i] == token]

            encoded_prompt.append(token)
            i += 1

        return encoded_prompt[answer_start:]


class FunctionParam:

    def __init__(self, manager: Manager, func_by_name: dict) -> None:

        self.manager = manager
        self.func_by_name = func_by_name

    def get_parameters(self, encoded_fn: list[int], user_prompt: str,
                       valid_int: list, valid_bool: list) -> dict:

        fn = self.manager.llm.decode(encoded_fn)
        encoded_prompt = self.prompt(user_prompt, fn)

        curr_data: dict[str, Any] = {}

        curr_data["prompt"] = user_prompt
        curr_data["name"] = fn
        curr_data["parameters"] = {}

        for key, value in self.func_by_name[fn]['params'].items():

            respond: list[int] = []
            encoded_prompt += self.manager.encode(str(key) + '=')
            flag = True

            if value.type in ["number", "integer"]:
                valid_token = valid_int

            elif value.type == 'boolean':
                valid_token = valid_bool

            else:
                valid_token = None

            while flag:

                logits = self.manager.llm.get_logits_from_input_ids(
                    encoded_prompt)

                if valid_token == valid_int:

                    if len(respond) > 40 and len(set(respond[-40:])) == 1:

                        respond.append(self.manager.encode(',')[0])
                        self.add_data(curr_data, key, value, respond)
                        break

                token = self.manager.constraint_decode(logits, valid_token)

                respond.append(token)
                encoded_prompt.append(token)

                if not self.check(token, value):

                    flag = False
                    self.add_data(curr_data, key, value, respond)

        return curr_data

    def prompt(self, user_prompt: str, fn: str) -> list[int]:

        prompt = (f"User request: {user_prompt}\nSelected Function: {fn}\n"
                  f"Function description: {self.func_by_name[fn]['descrp']}\n"
                  f"Function parameters: {self.func_by_name[fn]['params']}.\n"
                  "Read the parameter name then extract its value.\n")

        return self.manager.encode(prompt)

    def encoded_valid_token(self) -> tuple[list[int], list[int]]:

        interger = self.manager.encode(".01-234\n56789,")
        bools = self.manager.encode(" True False ,")

        return (interger, bools)

    def check(self, token: int, value: TypeLiteral) -> bool:

        last_token = self.manager.llm.decode(token)

        if value.type == 'string':
            return "\n" not in last_token

        return (("\n" not in last_token) and (',' not in last_token))

    def clean_answer(self, answer: list[int],
                     typee: TypeLiteral) -> Union[int, float, str]:

        dec_answer = self.manager.llm.decode(answer)

        if typee.type in ["number", "integer"]:

            if ',' in dec_answer:
                i = dec_answer.index(',')

            elif "\n" in dec_answer:
                i = dec_answer.index('\n')

            if typee.type == "integer":

                if '.' in dec_answer:
                    i2 = dec_answer.index('.')
                    i = i2 if i2 < i else i

                return int(dec_answer[:i])

            return float(dec_answer[:i])

        clean_answer = dec_answer.strip(
        ).strip(",").strip("'").strip('"').strip()

        return clean_answer

    def add_data(self, curr_data: dict, key: str, value: Any,
                 answer: list[int]) -> None:

        clean_answer = self.clean_answer(answer, value)
        curr_data["parameters"][key] = clean_answer
