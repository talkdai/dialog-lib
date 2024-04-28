.PHONY: test

test:
	poetry run pytest --cov=dialog_lib dialog_lib/tests/