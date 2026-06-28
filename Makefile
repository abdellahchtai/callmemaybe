funct_defs = data/input/functions_definition.json
input_prompt = data/input/function_calling_tests.json
output = data/output/output.txt

.PHONY: run install debug clean lint


run:
	@uv run -m src --functions_definition $(funct_defs) --input \
	$(input_prompt) --output $(output)

install:
	@uv sync

debug:
	@uv run -m pdb -m src --functions_definition $(funct_defs) --input \
	$(input_prompt) --output $(output)

clean:
	@rm -rf .mypy_cache src/__pycache__

lint:
	@flake8 src/
	@mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs

lint-strict:
	@flake8 src/
	@mypy src/ --strict
