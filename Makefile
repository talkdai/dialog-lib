.PHONY: test bump-beta bump-major bump-minor

test:
	poetry run pytest --cov=dialog_lib dialog_lib/tests/

bump-prepatch:
	poetry version --next-phase prepatch

bump-preminor:
	poetry version --next-phase preminor

bump-premajor:
	poetry version --next-phase premajor