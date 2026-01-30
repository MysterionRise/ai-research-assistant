# ARIA - AI Research Intelligence Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)

> **Enterprise-grade AI Research Assistant for Life Sciences and Materials Science R&D**

ARIA (AI Research Intelligence Assistant) is a comprehensive, cloud-first, AI-native platform designed to accelerate scientific discovery. Built for organizations like Corning, pharmaceutical companies, and biotech firms, ARIA transforms R&D operations from passive data management to **active AI collaboration**—evolving from a digital tool into an autonomous "coworker" capable of reasoning, planning, and execution within rigorous scientific constraints.

---

## 🎯 The Strategic Imperative

### The Recursive Value Loop

For materials science companies, AI creates a self-reinforcing strategic advantage:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Recursive AI Value Loop                          │
│                                                                     │
│  ┌──────────────┐     Drives Demand      ┌──────────────────────┐  │
│  │  AI/LLM      │ ──────────────────────▶│  Advanced Optical    │  │
│  │  Expansion   │                        │  Materials (5x more  │  │
│  │              │                        │  connectivity)       │  │
│  └──────────────┘                        └──────────────────────┘  │
│         ▲                                          │               │
│         │         Enables Faster                   ▼               │
│         │           Discovery         ┌──────────────────────┐    │
│         └─────────────────────────────│  AI-Powered R&D      │    │
│                                       │  (weeks vs. years)   │    │
│                                       └──────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Market Opportunity

| Metric | Value | Source |
|--------|-------|--------|
| AI-Assisted R&D Market | **$60-110 billion** | McKinsey |
| Life Sciences AI Adoption | **63%** of professionals | Industry Survey |
| AI vs Traditional Discovery | **18 months vs 4.5 years** | Insilico Medicine |
| Time Spent on Data Search | **70%+** of R&D time | Enterprise Studies |
| Reproducibility Crisis Cost | **$28 billion/year** | Preclinical Research |

### The Data Paradox in Science

Scientific industries face a unique challenge: massive data volumes yet acute data sparsity in high-value domains. While companies may have millions of data points on common compounds, data on novel glass compositions or rare ceramics is often limited to hundreds of experimental points. ARIA addresses this through:

1. **Liberating Unstructured Data**: Parsing tables, graphs, and complex layouts from PDFs
2. **Augmenting Sparse Data**: Physics-informed ML and synthetic data generation
3. **Connecting Silos**: Bridging wet lab (experiments) and dry lab (computational)

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Implementation Roadmap](#implementation-roadmap)
- [Compliance & Security](#compliance--security)
- [Contributing](#contributing)

---

## 🏗️ Architecture Overview

ARIA employs a modular, agentic architecture that rigorously separates **Reasoning (LLMs)** from **Domain Knowledge (Vector DBs)** and **Execution (Tools/APIs)**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARIA Platform                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Presentation Layer                               │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │   │
│  │  │  Web UI  │  │   API    │  │   CLI    │  │  Jupyter/Lab     │    │   │
│  │  │ (React)  │  │(FastAPI) │  │  (Rich)  │  │   Extensions     │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │              Agentic Orchestration Layer (LangGraph + AutoGen)       │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────┐      │   │
│  │  │  Planner  │  │ Executor  │  │  Critic   │  │   Memory    │      │   │
│  │  │  Agent    │  │  Agents   │  │  Agent    │  │   Manager   │      │   │
│  │  │           │  │           │  │           │  │             │      │   │
│  │  │ Decomposes│  │ Literature│  │ Validates │  │ Short-term  │      │   │
│  │  │ tasks     │  │ SQL Query │  │ physics/  │  │ Long-term   │      │   │
│  │  │           │  │ Simulation│  │ chemistry │  │ Entity      │      │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └─────────────┘      │   │
│  │                                                                      │   │
│  │  ReAct Pattern: Reason → Act → Observe → Reason...                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                    Model Zoo (Brains Layer)                          │   │
│  │  ┌─────────────────────────────────────────────────────────┐        │   │
│  │  │              Generalist LLMs (Reasoning)                 │        │   │
│  │  │  • Claude Sonnet 4 (primary)   • GPT-4o / o1            │        │   │
│  │  │  • Llama 3 70B (on-premise)    • Gemini Pro (2M ctx)    │        │   │
│  │  └─────────────────────────────────────────────────────────┘        │   │
│  │  ┌─────────────────────────────────────────────────────────┐        │   │
│  │  │              Domain-Specific Models                      │        │   │
│  │  │  • BioGPT (PubMed)       • ChemBERTa (molecules)        │        │   │
│  │  │  • PubMedBERT            • MolT5 (text↔SMILES)          │        │   │
│  │  │  • ESM-3 / AlphaFold     • MatBERT (materials)          │        │   │
│  │  │  • GNNs (crystals/glass) • CGCNN                        │        │   │
│  │  └─────────────────────────────────────────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                Scientific Multimodal RAG Pipeline                    │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐        │   │
│  │  │ Semantic │→│  Hybrid  │→│Cross-Enc │→│  Citation-   │        │   │
│  │  │ Chunking │  │  Search  │  │ Reranker │  │  Aware Gen   │        │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘        │   │
│  │                                                                      │   │
│  │  Dual Retriever: Text Embeddings + Molecular Embeddings (ChemRAG)    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                    Data & Storage Layer                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐      │   │
│  │  │  PostgreSQL  │  │    Redis     │  │      Vector DB       │      │   │
│  │  │  + pgvector  │  │    Cache     │  │  Pinecone/Weaviate   │      │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐      │   │
│  │  │ Elasticsearch│  │    Neo4j     │  │       S3/GCS         │      │   │
│  │  │  Full-text   │  │ Knowledge    │  │   Document Store     │      │   │
│  │  └──────────────┘  │    Graph     │  └──────────────────────┘      │   │
│  │                    └──────────────┘                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────▼───────────────────────────────────┐   │
│  │                    Integration Layer                                 │   │
│  │  • LIMS: Benchling, LabVantage    • Literature: PubMed, arXiv       │   │
│  │  • ELN: Signals Notebook          • Patents: USPTO, EPO             │   │
│  │  • Instruments: SiLA 2 protocol   • Semantic Scholar, Scopus        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              Compliance & Security (21 CFR Part 11 / GxP)            │   │
│  │  • Audit Trails      • RBAC           • Encryption (AES-256)        │   │
│  │  • E-Signatures      • Data Retention • Customer-Managed Keys       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Separation of Concerns**: Reasoning (LLMs) ↔ Knowledge (Vector DBs) ↔ Execution (Tools)
2. **Human-in-the-Loop**: AI recommends, humans approve (critical for GxP compliance)
3. **Code as Policy**: Deterministic calculations via code execution, not LLM arithmetic
4. **Grounded Generation**: All responses anchored to retrieved documents/data
5. **Multimodal First**: Native support for text, tables, molecular structures, spectra, images

---

## ✨ Key Features

### Phase 1: Literature Intelligence
- **Semantic Search**: Query 125M+ scientific papers across PubMed, arXiv, Semantic Scholar
- **Paper Summarization**: AI-powered synthesis with sentence-level citations
- **Systematic Review Automation**: 80% time reduction in literature reviews
- **Citation Network Exploration**: Discover related work and research gaps

### Phase 2: Data & Protocol Management  
- **SQL Agent**: Natural language queries to LIMS/ELN databases with schema awareness
- **Protocol Templates**: Version-controlled, audit-trailed experiment protocols
- **Multimodal PDF Parsing**: Extract tables, figures using VLMs (GPT-4V, pdfplumber)
- **Research Alerts**: Automated monitoring for new publications

### Phase 3: Advanced Scientific AI
- **Hypothesis Generation**: AI-suggested research directions based on literature
- **Property Prediction**: GNN-based molecular/material property prediction
- **Inverse Design**: Generate molecules/materials from natural language specs (MolT5)
- **Prior Art Analysis**: Automated patent landscape analysis

### Scientific Multimodal RAG vs Standard RAG

| Feature | Standard RAG | Scientific Multimodal RAG |
|---------|--------------|---------------------------|
| **Input** | Text only | Text + Tables + Chemical Structures (SMILES) + Graphs |
| **Chunking** | Fixed character count | Semantic (Section-based) + Molecule-based |
| **Retrieval** | Keyword/Semantic Similarity | Semantic + Substructure Search + Metadata Filters |
| **Output** | Text Summary | Text + Data Tables + Chemical Renderings |
| **Validation** | None | Physics/Chemistry Rule Checkers |

---

## 🛠️ Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|------------|---------|
| **API Framework** | FastAPI | Async REST API with OpenAPI docs |
| **Task Queue** | Celery + Redis | Background job processing |
| **Orchestration** | LangGraph + LangChain | Agentic workflow management |
| **Multi-Agent** | AutoGen | Complex multi-agent collaboration |
| **Database** | PostgreSQL + pgvector | Relational + vector storage |
| **Cache** | Redis | Session, rate limiting, caching |
| **Search** | Elasticsearch | Full-text + hybrid search |

### AI/ML Model Zoo
| Component | Technology | Purpose |
|-----------|------------|---------|
| **LLM (Cloud)** | Claude Sonnet 4, GPT-4o, Gemini | Primary reasoning engine |
| **LLM (On-Premise)** | Llama 3 70B (QLoRA fine-tuned) | IP protection, fixed costs |
| **Embeddings** | OpenAI, Cohere, BGE | Text vectorization |
| **Molecular** | ChemBERTa-2, MolT5 | Chemical/molecular AI |
| **Protein** | ESM-3, AlphaFold | Structure prediction |
| **Materials** | MatBERT, CGCNN, GNNs | Materials property prediction |
| **Knowledge Graph** | Neo4j | Biomedical relationships |

### Infrastructure
| Component | Technology | Purpose |
|-----------|---------|---------|
| **Cloud** | AWS (primary), GCP, Azure | Compute & storage |
| **Container** | Docker + Kubernetes | Orchestration |
| **CI/CD** | GitHub Actions | Automation |
| **Monitoring** | Prometheus + Grafana | Observability |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+ (for frontend)
- Docker & Docker Compose
- PostgreSQL 15+ with pgvector extension
- Redis 7+

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/aria-research-assistant.git
cd aria-research-assistant

# Install development dependencies
make install-dev

# Set up pre-commit hooks (REQUIRED - enforces all quality gates)
make setup-hooks

# Configure environment
cp .env.example .env
# Edit .env with your API keys and configuration

# Start development services
make dev-services

# Run database migrations
make migrate

# Start the development server
make dev
```

---

## 💻 Development Setup

### Code Quality Standards

This project enforces **strict code quality** through automated tooling. **All commits must pass these checks.**

| Tool | Purpose | Config |
|------|---------|--------|
| **Ruff** | Linting + formatting (replaces flake8, black, isort) | `pyproject.toml` |
| **MyPy** | Static type checking (strict mode) | `pyproject.toml` |
| **Pytest** | Testing framework (80% coverage minimum) | `pyproject.toml` |
| **Pre-commit** | Git hooks for quality gates | `.pre-commit-config.yaml` |
| **Bandit** | Security vulnerability scanning | `pyproject.toml` |

### Pre-commit Hooks

**⚠️ CRITICAL: All commits must pass quality checks. Never use `git commit --no-verify`.**

```bash
# Install hooks (REQUIRED before any development)
pre-commit install
pre-commit install --hook-type commit-msg

# Run all hooks manually
pre-commit run --all-files
```

### Development Commands

```bash
# Code Quality
make lint                 # Run Ruff linter with auto-fix
make format               # Format code with Ruff
make typecheck            # Run MyPy type checking
make security             # Run Bandit + Safety security scans
make check-all            # Run ALL checks (lint + type + test + security)

# Testing
make test                 # Run all tests
make test-cov             # Run tests with coverage (must be ≥80%)

# Development
make dev                  # Start development server
make dev-services         # Start Docker services
```

---

## 📁 Project Structure

```
aria-research-assistant/
├── .github/workflows/         # CI/CD pipelines
├── docs/                      # Documentation
├── src/aria/
│   ├── api/                   # FastAPI application
│   ├── agents/                # Agentic system (Planner, Executor, Critic)
│   ├── models/                # LLM and domain model integrations
│   ├── rag/                   # Scientific RAG pipeline
│   ├── data/                  # Ingestion, connectors, storage
│   ├── compliance/            # Audit trails, signatures, validation
│   └── integrations/          # LIMS, ELN integrations
├── tests/                     # Unit, integration, e2e tests
├── infrastructure/            # Docker, K8s, Terraform
├── .pre-commit-config.yaml    # Quality gate hooks
├── pyproject.toml             # Project configuration
├── Makefile                   # Development commands
└── CLAUDE.md                  # Claude Code instructions
```

---

## 📅 Implementation Roadmap

### Phase 1: PoC (Months 1-3)
- Project setup with CI/CD and quality gates
- Basic RAG pipeline with PDF ingestion
- Literature search (PubMed, arXiv)
- **Success**: 80% accuracy on "Golden Set" questions

### Phase 2: Pilot (Months 4-6)
- SQL Agent for LIMS integration
- ChemBERTa molecular search
- Multi-agent system (Planner-Executor-Critic)
- **Success**: RAG faithfulness >0.8

### Phase 3: MVP (Months 7-12)
- Full LIMS read/write integration
- 21 CFR Part 11 compliance
- Domain model zoo (AlphaFold, GNNs)
- **Success**: Production deployment, 99.9% uptime

---

## 🔒 Compliance & Security

### 21 CFR Part 11 / GxP Requirements

| Requirement | Implementation |
|-------------|----------------|
| **Audit Trails** | Every interaction logged with full context |
| **E-Signatures** | Two-factor authentication |
| **Reproducibility** | Fixed model seeds, frozen versions |
| **Access Controls** | Role-based permissions (RBAC) |

### Hallucination Mitigation

In science, hallucination is a **critical liability**. ARIA employs:

1. **Grounding**: All responses anchored to retrieved documents
2. **Code as Policy**: Calculations executed as Python, not LLM arithmetic
3. **Critic Agent**: Adversarial review against physics/chemistry rules
4. **Confidence Calibration**: Human routing for low-confidence outputs

---

## 🤝 Contributing

1. Create a feature branch from `main`
2. **Ensure all pre-commit hooks pass** (required)
3. Submit a pull request
4. Code review and approval required

### Commit Convention

```
feat: add molecular similarity search
fix: correct citation formatting
docs: update API documentation
test: add RAG pipeline tests
```

---

## 📄 License

Copyright © 2024-2025. All rights reserved. Proprietary and confidential.

---

<p align="center">
  <strong>ARIA</strong> - Transforming Data Archives into Engines of Discovery
</p>
