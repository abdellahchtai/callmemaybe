import numpy as np


class FunctionParam:

    def __init__(self, manager, func_by_name):

        self.manager = manager
        self.func_by_name = func_by_name

    def get_parameters(self, encoded_fn: list[int], user_prompt: str,
                       valid_int, valid_bool):

        fn = self.manager.llm.decode(encoded_fn)
        encoded_prompt = self.prompt(user_prompt, fn)

        curr_data = {}

        curr_data["prompt"] = user_prompt
        curr_data["name"] = fn
        curr_data["parameters"] = {}

        for key, value in self.func_by_name[fn]['params'].items():

            respond = []
            encoded_prompt += self.manager.encode(str(key) + '=')
            flag = True

            if value.type in ["number", "integer"]:
                valid_token = valid_int

            elif value.type == 'boolean':
                valid_token = valid_bool

            else:
                print(value.type)
                valid_token = True

            while flag:

                logits = self.manager.llm.get_logits_from_input_ids(
                    encoded_prompt)

                if valid_token == valid_int:

                    if len(respond) > 10 and len(set(respond[-10:])) == 1:

                        respond.append(self.manager.encode(',')[0])
                        self.add_data(curr_data, key, value, respond)
                        break

                if valid_token is True:
                    token = np.argmax(logits)

                else:
                    token = self.manager.constraint_decode(logits, valid_token)

                respond.append(token)
                encoded_prompt.append(token)
                print(self.manager.llm.decode(encoded_prompt))

                if not self.check(token):

                    flag = False
                    self.add_data(curr_data, key, value, respond)

        return curr_data

    def prompt(self, user_prompt, fn):

        prompt = (f"User request: {user_prompt}\nSelected Function: {fn}\n"
                  f"Function description: {self.func_by_name[fn]['descrp']}\n"
                  f"Function parameters: {self.func_by_name[fn]['params']}.\n"
                  "Read the parameter name then extract its value.\n")

        return self.manager.encode(prompt)

    def encoded_valid_token(self):

        interger = self.manager.encode(".01-234\n56789,")
        bools = self.manager.encode(" True False ,")

        return interger, bools

    def check(self, token):

        last_token = self.manager.llm.decode(token)
        return (("\n" not in last_token) and (',' not in last_token))

    def clean_answer(self, answer: list[int], typee) -> str:

        dec_answer = self.manager.llm.decode(answer)

        if typee.type in ["number", "integer"]:

            if ',' in dec_answer:
                i = dec_answer.index(',')

            elif "\n" in dec_answer:
                i = dec_answer.index('\n')

            return float(dec_answer[:i])

        clean_answer = dec_answer.strip(
        ).strip(",").strip("'").strip('"').strip()

        return clean_answer

    def add_data(self, curr_data: dict, key, value, answer) -> None:

        clean_answer = self.clean_answer(answer, value)
        curr_data["parameters"][key] = clean_answer
