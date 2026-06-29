import argparse
import json
from pydantic import BaseModel, ConfigDict, TypeAdapter, Field
from typing import Literal, Dict


class ParseArgument:

    def __init__(self) -> None:

        self.args = ParseArgument.arg_parse()
        self.set_default()

    @staticmethod
    def arg_parse() -> argparse.Namespace:

        parse = argparse.ArgumentParser()

        parse.add_argument(
            '--functions_definition',
            help='Path to function definition file.')
        parse.add_argument('--input', help='Path to input file.')
        parse.add_argument('--output', help='Path to output file.')

        return parse.parse_args()

    def set_default(self):

        if self.args.input is None:
            self.args.input = '/data/input/function_calling_tests.json'

        if self.args.functions_definition is None:
            self.args.functions_definiton = ('/data/input/functions_d'
                                             'efinition.json')

        if self.args.output is None:
            self.args.output = '/data/output/ffunction_calls.json'


class ValidPrompt(BaseModel):

    model_config = ConfigDict(extra='forbid')

    prompt: str


class TypeLiteral(BaseModel):

    type: Literal["string", "number", "boolean", "integer"
                  ] = Field(min_length=1)


class FunctionDef(BaseModel):

    model_config = ConfigDict(extra='forbid')

    name: str = Field(min_length=2)
    description: str = Field(min_length=5)
    parameters: Dict[str, TypeLiteral] = Field(min_length=1)
    returns: TypeLiteral


class ParseFiles:

    @staticmethod
    def prompts_file(input_file1: str) -> list[ValidPrompt]:

        prompts = []

        with open(input_file1) as f:

            prompts_dict = json.load(f)
            adapter = TypeAdapter(list[ValidPrompt])

            prompts = adapter.validate_python(prompts_dict)

        return prompts

    @staticmethod
    def definition_file(input_file2: str) -> list[FunctionDef]:

        with open(input_file2) as f:

            functs_def_dict = json.load(f)
            adapter = TypeAdapter(list[FunctionDef])
            functs_def = adapter.validate_python(functs_def_dict)

        return functs_def
