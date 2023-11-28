default: format run

format:
	poetry run black src/**/*.py
	poetry run black src/*.py
	poetry run black main.py

run:
	poetry run python3 main.py
