"""Unit tests for protocols endpoints."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient


class TestProtocolModels:
    """Tests for protocol request/response models."""

    def test_protocol_step_creation(self) -> None:
        """Test ProtocolStep model creation."""
        from aria.api.routes.protocols import ProtocolStep

        step = ProtocolStep(
            step_number=1,
            title="Mix reagents",
            description="Combine reagent A and B",
        )

        assert step.step_number == 1
        assert step.title == "Mix reagents"
        assert step.equipment == []
        assert step.reagents == []

    def test_protocol_step_with_all_fields(self) -> None:
        """Test ProtocolStep with all fields."""
        from aria.api.routes.protocols import ProtocolStep

        step = ProtocolStep(
            step_number=2,
            title="Heat mixture",
            description="Heat to 100C",
            duration_minutes=30,
            equipment=["Hot plate", "Thermometer"],
            reagents=["Water"],
            safety_notes=["Use heat resistant gloves"],
            parameters={"temperature": 100, "unit": "C"},
        )

        assert step.duration_minutes == 30
        assert len(step.equipment) == 2
        assert step.parameters["temperature"] == 100

    def test_protocol_create_request(self) -> None:
        """Test ProtocolCreateRequest model."""
        from aria.api.routes.protocols import ProtocolCreateRequest, ProtocolStep

        request = ProtocolCreateRequest(
            name="Test Protocol",
            description="A test protocol",
            steps=[
                ProtocolStep(
                    step_number=1,
                    title="Step 1",
                    description="Do something",
                )
            ],
            tags=["test", "lab"],
        )

        assert request.name == "Test Protocol"
        assert len(request.steps) == 1
        assert request.tags == ["test", "lab"]

    def test_protocol_generate_request(self) -> None:
        """Test ProtocolGenerateRequest model."""
        from aria.api.routes.protocols import ProtocolGenerateRequest

        request = ProtocolGenerateRequest(
            objective="Extract DNA from plant cells for analysis",
            constraints=["Use only cold reagents"],
            available_equipment=["Centrifuge", "PCR machine"],
            safety_level="elevated",
        )

        assert "DNA" in request.objective
        assert len(request.constraints) == 1
        assert request.safety_level == "elevated"

    def test_protocol_list_response(self) -> None:
        """Test ProtocolListResponse model."""
        from aria.api.routes.protocols import ProtocolListResponse

        response = ProtocolListResponse(
            total=0,
            protocols=[],
            page=1,
            page_size=20,
        )

        assert response.total == 0
        assert response.page == 1


class TestListProtocolsEndpoint:
    """Tests for list protocols endpoint."""

    @pytest.mark.asyncio
    async def test_list_protocols_returns_empty_list(self) -> None:
        """Test that list protocols returns empty list initially."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/protocols")

            assert response.status_code == 200
            data = response.json()
            assert data["total"] == 0
            assert data["protocols"] == []
            assert data["page"] == 1

    @pytest.mark.asyncio
    async def test_list_protocols_with_pagination(self) -> None:
        """Test list protocols with pagination parameters."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/v1/protocols?page=2&page_size=10")

            assert response.status_code == 200
            data = response.json()
            assert data["page"] == 2
            assert data["page_size"] == 10


class TestGetProtocolEndpoint:
    """Tests for get protocol endpoint."""

    @pytest.mark.asyncio
    async def test_get_protocol_not_found(self) -> None:
        """Test that get protocol returns 404 for unknown ID."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            protocol_id = uuid4()
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.get(f"/api/v1/protocols/{protocol_id}")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"]


class TestCreateProtocolEndpoint:
    """Tests for create protocol endpoint."""

    @pytest.mark.asyncio
    async def test_create_protocol_success(self) -> None:
        """Test creating a protocol successfully."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/api/v1/protocols",
                    json={
                        "name": "New Protocol",
                        "description": "Test description",
                        "steps": [
                            {
                                "step_number": 1,
                                "title": "First step",
                                "description": "Do this first",
                            }
                        ],
                    },
                )

            assert response.status_code == 201
            data = response.json()
            assert data["name"] == "New Protocol"
            assert data["status"] == "draft"
            assert data["version"] == "1.0"


class TestGenerateProtocolEndpoint:
    """Tests for generate protocol endpoint."""

    @pytest.mark.asyncio
    async def test_generate_protocol_returns_draft(self) -> None:
        """Test that generate protocol returns a draft."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/api/v1/protocols/generate",
                    json={
                        "objective": "Extract proteins from E. coli cells for analysis",
                    },
                )

            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "draft"
            assert "ai-generated" in data["tags"]


class TestUpdateProtocolEndpoint:
    """Tests for update protocol endpoint."""

    @pytest.mark.asyncio
    async def test_update_protocol_not_found(self) -> None:
        """Test that update returns 404 for unknown protocol."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            protocol_id = uuid4()
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.put(
                    f"/api/v1/protocols/{protocol_id}",
                    json={
                        "name": "Updated Protocol",
                        "description": "Updated description",
                        "steps": [
                            {
                                "step_number": 1,
                                "title": "Step",
                                "description": "Description",
                            }
                        ],
                    },
                )

            assert response.status_code == 404


class TestApproveProtocolEndpoint:
    """Tests for approve protocol endpoint."""

    @pytest.mark.asyncio
    async def test_approve_protocol_not_found(self) -> None:
        """Test that approve returns 404 for unknown protocol."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            protocol_id = uuid4()
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.post(f"/api/v1/protocols/{protocol_id}/approve")

            assert response.status_code == 404


class TestArchiveProtocolEndpoint:
    """Tests for archive protocol endpoint."""

    @pytest.mark.asyncio
    async def test_archive_protocol_not_found(self) -> None:
        """Test that archive returns 404 for unknown protocol."""
        from aria.api.app import create_app

        with (
            patch("aria.api.app.init_db", new_callable=AsyncMock),
            patch("aria.api.app.close_db", new_callable=AsyncMock),
        ):
            app = create_app()

            protocol_id = uuid4()
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                response = await client.delete(f"/api/v1/protocols/{protocol_id}")

            assert response.status_code == 404
