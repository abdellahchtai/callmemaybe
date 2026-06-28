from .parsing import ParseArgument, ParseFiles
import json
from pydantic import ValidationError
import time


if __name__ == "__main__":

    start = time.perf_counter()
    argv = ParseArgument()

    try:

        from .manager import Manager

        prompts = ParseFiles.prompts_file(argv.args.input)
        functs_def = ParseFiles.definition_file(argv.args.functions_definition)

        if not prompts:
            raise ValueError("Error: The prompts file doesn't have valid "
                             "data.")

        if not functs_def:
            raise ValueError("Error: The function definition file doesn't "
                             "have valid data.")

        model_name1 = "Qwen/Qwen3-0.6B"
        model_name2 = "Qwen/Qwen2.5-1.5B-Instruct"
        model = Manager(model_name1)
        model.main(prompts, functs_def)

    except ValidationError as e:

        print(
                f"Error: {e.errors()[0]['msg']}. input="
                f"'{e.errors()[0]['input']}'")

    except (FileNotFoundError, PermissionError, IsADirectoryError,
            json.decoder.JSONDecodeError, ValueError, NotADirectoryError,
            KeyboardInterrupt)as e:

        print(e)

    end = time.perf_counter()
    print(f"This code take {(end - start) / 60:.2f} second.")
