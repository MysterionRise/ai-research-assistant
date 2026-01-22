# CLAUDE.md - AI Assistant Guidelines

This document provides essential context for AI assistants working with this codebase.

## Project Overview

**semantic-search-cookbook** is a collection of recipes and examples for implementing semantic search applications. The project is in early development phase with foundational tooling configured.

**License:** MIT (Copyright 2023 Konstantin Perikov)

## Current Repository Structure

```
/
├── .gitignore                 # Git ignore rules (.idea)
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── LICENSE                    # MIT license
├── README.md                  # Project description
├── requirements-dev.txt       # Development dependencies
└── CLAUDE.md                  # This file
```

## Technology Stack

- **Language:** Python 3
- **Code Formatter:** Black (79-char line length)
- **Import Sorter:** isort (Black profile)
- **Linter:** flake8
- **Commit Validation:** commitizen

## Development Setup

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Set Up Pre-commit Hooks

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

## Code Style Guidelines

### Formatting Rules

- **Line length:** 79 characters (enforced by Black and isort)
- **Import sorting:** Use isort with Black profile
- **Code formatting:** Black formatter

### Linting

flake8 is configured with:
- Max line length: 120 (for flexibility with long strings/comments)
- Ignored warnings: W605 (invalid escape sequence), E203 (whitespace before ':')

### Example Python File Structure

```python
"""Module docstring."""

# Standard library imports
import os
import sys

# Third-party imports
import numpy as np

# Local imports
from .utils import helper


def function_name():
    """Function docstring."""
    pass
```

## Commit Message Convention

This project uses **commitizen** for standardized commit messages. Follow the conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no code change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

### Examples

```
feat(search): add vector similarity search function
fix(embeddings): handle empty input gracefully
docs(readme): update installation instructions
chore: update pre-commit hooks versions
```

## Pre-commit Hooks

The following hooks run automatically on every commit:

1. **isort** - Sorts imports (Black-compatible)
2. **black** - Formats Python code
3. **flake8** - Lints Python code
4. **commitizen** - Validates commit messages (on commit-msg stage)
5. **check-yaml** - Validates YAML syntax
6. **end-of-file-fixer** - Ensures files end with newline
7. **trailing-whitespace** - Removes trailing whitespace

**Note:** `fail_fast: true` is enabled - hooks stop on first failure.

## Guidelines for AI Assistants

### When Writing Python Code

1. Keep lines under 79 characters
2. Use Black-compatible formatting
3. Sort imports with isort (standard lib, third-party, local)
4. Add docstrings to modules, classes, and functions
5. Follow PEP 8 naming conventions

### When Making Commits

1. Use conventional commit format (commitizen)
2. Keep subject line concise (under 50 chars ideal)
3. Use imperative mood ("add" not "added")
4. Reference issues when applicable

### When Adding New Files

1. Python files: Include module docstring
2. Configuration files: Follow existing patterns
3. Add appropriate entries to .gitignore if needed

### When Creating New Features

Consider creating recipe-style documentation that includes:
- Problem description
- Solution approach
- Code example
- Usage instructions

## Future Development Expectations

Based on project goals, expect development of:
- Semantic search implementations
- Vector embedding utilities
- Search index integrations
- Example notebooks/scripts

## Quick Reference Commands

```bash
# Format code
black --line-length=79 .

# Sort imports
isort --profile=black --line-length=79 .

# Run linter
flake8 --max-line-length=120 .

# Run all pre-commit hooks
pre-commit run --all-files

# Validate commit message
echo "feat: add feature" | commitizen check
```
