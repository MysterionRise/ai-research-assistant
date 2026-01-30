.PHONY: help install install-dev setup-hooks dev test lint format typecheck security check-all

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install all dependencies (production + dev)
	pip install -e ".[dev]"

setup-hooks: ## Install pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

dev: ## Start development server
	uvicorn aria.api.app:app --reload --host 0.0.0.0 --port 8000

dev-services: ## Start Docker services
	docker-compose -f infrastructure/docker/docker-compose.yml up -d postgres redis

lint: ## Run Ruff linter
	ruff check src tests --fix

format: ## Format code
	ruff format src tests

typecheck: ## Run MyPy
	mypy src

test: ## Run tests
	pytest tests -v

test-cov: ## Run tests with coverage
	pytest tests --cov=src/aria --cov-report=term-missing --cov-fail-under=80

security: ## Run security scans
	bandit -c pyproject.toml -r src
	safety check

check-all: lint typecheck test security ## Run all checks
	@echo "All checks passed!"

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage

migrate: ## Run database migrations
	alembic upgrade head

migrate-create: ## Create new migration
	alembic revision --autogenerate -m "$(msg)"

worker: ## Start Celery worker
	celery -A aria.worker.celery_app worker --loglevel=info

test-unit: ## Run unit tests only
	pytest tests/unit -v

test-integration: ## Run integration tests
	pytest tests/integration -v

evaluate: ## Run RAG evaluation
	python -m aria.evaluation.ragas_eval --golden-set tests/fixtures/golden_set/literature_qa.json
