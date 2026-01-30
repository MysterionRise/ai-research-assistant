# CLAUDE.md - AI Research Assistant Development Guide

This file provides context and instructions for Claude Code when working on ARIA.

---

## Project Overview

ARIA is an enterprise-grade AI Research Assistant for Life Sciences and Materials Science R&D. It transforms passive data management into active AI collaboration through agentic workflows.

**Target Users**: Scientists, researchers, R&D teams at companies like Corning (materials), pharma, biotech.

**Key Differentiators**:
- Scientific multimodal RAG (text + tables + molecular structures)
- Agentic workflows with human-in-the-loop compliance
- 21 CFR Part 11 / GxP regulatory compliance
- On-premise deployment option for IP protection

---

## Development Commands

```bash
# Setup
make install-dev          # Install all dependencies
make setup-hooks          # Install pre-commit hooks (REQUIRED)

# Development
make dev                  # Start development server
make dev-services         # Start Docker services (PostgreSQL, Redis)

# Quality Checks (MUST PASS BEFORE COMMIT)
make lint                 # Run ruff linter
make format               # Format code with ruff
make typecheck            # Run mypy type checking
make test                 # Run pytest
make test-cov             # Run tests with coverage (≥80% required)
make security             # Run bandit security scan
make check-all            # Run ALL checks

# Database
make migrate              # Run database migrations
```

### Pre-commit Hooks

**CRITICAL**: All commits must pass pre-commit hooks. Never use `git commit --no-verify`.

Pre-commit runs automatically on `git commit`. If it fails:
1. Review the errors
2. Fix issues (many are auto-fixed)
3. Re-stage: `git add .`
4. Commit again

---

## Code Style & Standards

### Python Style (Enforced by Ruff)

```python
# ✅ GOOD: Type hints required for all functions
def search_literature(
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[Document]:
    """Search scientific literature.
    
    Args:
        query: Search query string
        limit: Maximum results to return
        filters: Optional filters
        
    Returns:
        List of matching documents
        
    Raises:
        SearchError: If search service unavailable
    """
    ...

# ❌ BAD: No type hints, no docstring
def search_literature(query, limit=10, filters=None):
    ...
```

### Import Order

```python
# Standard library
import json
from pathlib import Path

# Third-party
import httpx
from fastapi import APIRouter

# Local
from aria.agents.base import BaseAgent
from aria.config.settings import settings
```

### Naming Conventions

```python
# Classes: PascalCase
class LiteratureSearchAgent:
    pass

# Functions/methods: snake_case
def extract_citations(text: str) -> list[Citation]:
    pass

# Constants: SCREAMING_SNAKE_CASE
MAX_CONTEXT_LENGTH = 128000
DEFAULT_MODEL = "claude-sonnet-4-20250514"
```

### Error Handling

```python
# ✅ GOOD: Specific exceptions with context
from aria.exceptions import SearchError

try:
    results = await search_service.query(query)
except httpx.TimeoutException as e:
    logger.error("Search timeout", query=query, error=str(e))
    raise SearchError(f"Search timed out: {query}") from e

# ❌ BAD: Bare except
try:
    results = search_service.query(query)
except:
    return []
```

---

## Architecture Guidelines

### Agent Design Pattern (ReAct)

All agents follow the ReAct (Reasoning + Acting) pattern:

```python
from aria.agents.base import BaseAgent, AgentResponse, AgentState

class LiteratureAgent(BaseAgent):
    """Agent for literature search and synthesis."""
    
    name = "literature_agent"
    description = "Searches and synthesizes scientific literature"
    
    tools = [
        SearchPubMedTool(),
        SearchSemanticScholarTool(),
        SummarizePaperTool(),
    ]
    
    async def plan(self, state: AgentState) -> list[str]:
        """Decompose request into subtasks."""
        ...
    
    async def execute(self, state: AgentState) -> AgentResponse:
        """Execute the plan using tools."""
        ...
    
    async def critique(self, response: AgentResponse) -> AgentResponse:
        """Review and validate the response against physical laws."""
        ...
```

### RAG Pipeline Structure

```python
class ScientificRAGPipeline:
    """Multimodal RAG for scientific documents."""
    
    def __init__(
        self,
        chunker: ScientificChunker,
        embedder: HybridEmbedder,
        retriever: HybridRetriever,  # Text + Molecular
        reranker: CrossEncoderReranker,
        synthesizer: CitationAwareSynthesizer,
    ):
        ...
    
    async def query(
        self,
        question: str,
        filters: SearchFilters | None = None,
    ) -> RAGResponse:
        # 1. Retrieve candidates (hybrid: semantic + substructure)
        candidates = await self.retriever.search(question, filters)
        
        # 2. Rerank with cross-encoder
        reranked = await self.reranker.rerank(question, candidates)
        
        # 3. Synthesize with citations
        response = await self.synthesizer.generate(question, reranked)
        
        return response
```

### Hallucination Mitigation (Critical for Science)

```python
# ✅ Code as Policy: Execute calculations, don't ask LLM
async def calculate_molecular_weight(smiles: str) -> float:
    """Use rdkit for deterministic calculation."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Descriptors.MolWt(mol)

# ❌ Never ask LLM to calculate
response = await llm.complete("Calculate molecular weight of C6H12O6")
```

---

## Testing Standards

### Test Structure

```python
class TestLiteratureAgent:
    """Tests for LiteratureAgent."""
    
    @pytest.fixture
    def agent(self):
        return LiteratureAgent()
    
    async def test_plan_decomposes_query(self, agent, sample_state):
        """Test that plan() breaks query into subtasks."""
        plan = await agent.plan(sample_state)
        
        assert len(plan) > 0
        assert any("search" in step.lower() for step in plan)
```

### Coverage Requirements

- **Unit tests**: Minimum 80% coverage
- **Integration tests**: All API endpoints
- **E2E tests**: Critical user workflows

---

## Security Guidelines

### Secrets Management

```python
# ✅ GOOD: Use environment variables
from aria.config.settings import settings
client = OpenAI(api_key=settings.openai_api_key.get_secret_value())

# ❌ BAD: Hardcoded secrets
client = OpenAI(api_key="sk-...")
```

### Audit Logging (21 CFR Part 11)

```python
from aria.compliance.audit import audit_log

@audit_log(action="document.create")
async def create_document(doc: DocumentCreate, user: User) -> Document:
    """Create document with audit trail."""
    ...
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/aria/main.py` | Application entry point |
| `src/aria/config/settings.py` | Pydantic settings |
| `src/aria/api/app.py` | FastAPI application |
| `src/aria/agents/orchestrator.py` | Agent orchestration |
| `src/aria/rag/pipeline.py` | RAG implementation |
| `pyproject.toml` | Project config, linting, testing |
| `.pre-commit-config.yaml` | Git hooks |

---

## Common Tasks

### Adding a New Agent

1. Create `src/aria/agents/my_agent.py`
2. Inherit from `BaseAgent`
3. Implement `plan()`, `execute()`, `critique()`
4. Register in `src/aria/agents/__init__.py`
5. Add tests in `tests/unit/test_agents/`

### Adding a New API Endpoint

1. Create route in `src/aria/api/routes/`
2. Define Pydantic schemas
3. Register router in `src/aria/api/app.py`
4. Add tests

### Adding a Data Connector

1. Create connector in `src/aria/data/connectors/`
2. Implement: `search()`, `fetch()`, `transform()`
3. Add rate limiting and error handling
4. Add integration tests

---

## Troubleshooting

### Pre-commit Failures

```bash
# Ruff errors
ruff check --fix .
ruff format .

# MyPy errors
mypy src/  # See detailed errors

# Test failures
pytest -xvs  # Verbose, stop on first failure
```

### Docker Issues

```bash
docker-compose build --no-cache
docker-compose logs -f api
```
