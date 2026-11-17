.PHONY: setup generate-data pipeline train evaluate serve ui mlflow-ui test lint docker-up docker-down demo
export PYTHONPATH := src

setup:
	pip install -r requirements.txt

generate-data:
	python -m clinical_mlops.data_pipeline.generate_synthetic_docs

pipeline:
	python -m clinical_mlops.data_pipeline.pipeline

train:
	python -m clinical_mlops.models.train

evaluate:
	python -m clinical_mlops.models.evaluate

serve:
	uvicorn clinical_mlops.serving.api:app --reload --port 8000

ui:
	streamlit run apps/streamlit_app.py

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	pytest -v

lint:
	ruff check src tests apps

docker-up:
	docker compose up --build

docker-down:
	docker compose down

demo: pipeline train evaluate
	@echo "Demo complete. Run 'make serve' and 'make ui' in separate terminals to try the API / Streamlit tools."
