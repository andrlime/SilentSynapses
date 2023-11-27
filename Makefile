default: format run

format:
	poetry run black */**/*.py

run:
	poetry run python3 main.py