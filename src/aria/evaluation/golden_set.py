"""Golden set management for RAG evaluation."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EvalCase:
    """A single evaluation case for RAG evaluation."""

    id: str
    query: str
    expected_answer: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    metadata: dict = field(default_factory=dict)


@dataclass
class GoldenSet:
    """Collection of test cases for RAG evaluation."""

    name: str
    description: str
    test_cases: list[EvalCase] = field(default_factory=list)
    version: str = "1.0"

    @classmethod
    def from_json(cls, path: Path | str) -> "GoldenSet":
        """Load golden set from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            GoldenSet instance.
        """
        with open(path) as f:
            data = json.load(f)

        test_cases = [
            EvalCase(
                id=tc.get("id", str(i)),
                query=tc["query"],
                expected_answer=tc.get("expected_answer"),
                expected_sources=tc.get("expected_sources", []),
                category=tc.get("category", "general"),
                difficulty=tc.get("difficulty", "medium"),
                metadata=tc.get("metadata", {}),
            )
            for i, tc in enumerate(data.get("test_cases", []))
        ]

        return cls(
            name=data.get("name", "Golden Set"),
            description=data.get("description", ""),
            test_cases=test_cases,
            version=data.get("version", "1.0"),
        )

    def to_json(self, path: Path | str) -> None:
        """Save golden set to JSON file.

        Args:
            path: Output path.
        """
        data = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "test_cases": [
                {
                    "id": tc.id,
                    "query": tc.query,
                    "expected_answer": tc.expected_answer,
                    "expected_sources": tc.expected_sources,
                    "category": tc.category,
                    "difficulty": tc.difficulty,
                    "metadata": tc.metadata,
                }
                for tc in self.test_cases
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("golden_set_saved", path=str(path), count=len(self.test_cases))

    def filter_by_category(self, category: str) -> "GoldenSet":
        """Filter test cases by category.

        Args:
            category: Category to filter by.

        Returns:
            New GoldenSet with filtered cases.
        """
        filtered = [tc for tc in self.test_cases if tc.category == category]
        return GoldenSet(
            name=f"{self.name} ({category})",
            description=self.description,
            test_cases=filtered,
            version=self.version,
        )

    def filter_by_difficulty(self, difficulty: str) -> "GoldenSet":
        """Filter test cases by difficulty.

        Args:
            difficulty: Difficulty level (easy, medium, hard).

        Returns:
            New GoldenSet with filtered cases.
        """
        filtered = [tc for tc in self.test_cases if tc.difficulty == difficulty]
        return GoldenSet(
            name=f"{self.name} ({difficulty})",
            description=self.description,
            test_cases=filtered,
            version=self.version,
        )


def create_sample_golden_set() -> GoldenSet:
    """Create a sample golden set for testing.

    Returns:
        Sample GoldenSet with literature QA test cases.
    """
    test_cases = [
        EvalCase(
            id="lit_1",
            query="What are the main mechanisms of CRISPR-Cas9 gene editing?",
            expected_answer="CRISPR-Cas9 uses a guide RNA to direct the Cas9 nuclease to a specific DNA sequence, where it creates a double-strand break. The cell's repair mechanisms then either disrupt the gene (NHEJ) or insert new genetic material (HDR).",
            category="molecular_biology",
            difficulty="medium",
        ),
        EvalCase(
            id="lit_2",
            query="How does mRNA vaccine technology work?",
            expected_answer="mRNA vaccines deliver synthetic messenger RNA that encodes viral proteins. Cells use this mRNA to produce the viral protein, triggering an immune response without causing infection.",
            category="immunology",
            difficulty="medium",
        ),
        EvalCase(
            id="lit_3",
            query="What is the difference between Type 1 and Type 2 diabetes?",
            expected_answer="Type 1 diabetes is an autoimmune condition where the immune system attacks insulin-producing beta cells. Type 2 diabetes involves insulin resistance and progressive beta cell dysfunction.",
            category="endocrinology",
            difficulty="easy",
        ),
        EvalCase(
            id="mat_1",
            query="What are the properties of graphene that make it useful for electronics?",
            expected_answer="Graphene has exceptional electrical conductivity, high electron mobility, mechanical strength, and flexibility, making it promising for flexible electronics, transistors, and sensors.",
            category="materials_science",
            difficulty="medium",
        ),
        EvalCase(
            id="mat_2",
            query="How do lithium-ion batteries work?",
            expected_answer="Lithium-ion batteries store energy by moving lithium ions between the anode and cathode through an electrolyte during charge and discharge cycles.",
            category="materials_science",
            difficulty="easy",
        ),
    ]

    return GoldenSet(
        name="ARIA Literature QA Golden Set",
        description="Test cases for evaluating scientific literature question answering",
        test_cases=test_cases,
        version="1.0",
    )
