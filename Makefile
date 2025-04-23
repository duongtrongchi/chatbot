run:
	poetry run streamlit run src/ui.py --server.headless true

setup:
	pip install poetry
	poetry install
