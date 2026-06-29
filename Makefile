functions_definition = data/input/functions_definition.json
input = data/input/function_calling_tests.json
output = data/output/function_calls.json

.PHONY: run install debug clean lint


run:
	@uv run -m src --functions_definition $(functions_definition) --input \
	$(input) --output $(output)

install:
	@uv sync

debug:
	@uv run -m pdb -m src --functions_definition $(funct_defs) --input \
	$(input) --output $(output)

clean:
	@rm -rf .mypy_cache src/__pycache__

lint:
	@flake8 src/
	@mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports \
	--disallow-untyped-defs --check-untyped-defs
