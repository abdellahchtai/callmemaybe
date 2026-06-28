class FunctionName:

    def __init__(self, encoded_fn: list[list[int]],
                 avail_fts: list[int], manager) -> None:

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

            to_keep = {elm[i] for elm in encoded_functions}
            logits = self.manager.llm.get_logits_from_input_ids(encoded_prompt)
            token = self.manager.constraint_decode(
                logits, to_keep)

            encoded_functions = [elm for elm in encoded_functions
                                 if elm[i] == token]

            encoded_prompt.append(token)
            i += 1

        return encoded_prompt[answer_start:]
