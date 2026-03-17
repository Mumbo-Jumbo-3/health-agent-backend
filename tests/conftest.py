from pathlib import Path

import pytest

from health_agent.config import Settings


@pytest.fixture
def tmp_resources(tmp_path: Path) -> Path:
    resources = tmp_path / "resources"
    resources.mkdir()

    (resources / "nutrition.md").write_text(
        "# Nutrition Guide\n\n"
        "Eating a balanced diet rich in fruits, vegetables, whole grains, "
        "and lean proteins is essential for good health. Aim for at least "
        "5 servings of fruits and vegetables per day.\n\n"
        "## Hydration\n\n"
        "Drink at least 8 glasses of water daily to stay properly hydrated."
    )

    (resources / "exercise.txt").write_text(
        "Regular physical activity is one of the most important things you can do "
        "for your health. Adults should aim for at least 150 minutes of moderate "
        "aerobic activity per week, along with muscle-strengthening activities "
        "on 2 or more days per week."
    )

    return resources


@pytest.fixture
def test_settings(tmp_path: Path, tmp_resources: Path) -> Settings:
    return Settings(
        resources_dir=tmp_resources,
        chroma_persist_dir=tmp_path / ".chroma_db",
        openai_api_key="test-key",
        chunk_size=200,
        chunk_overlap=50,
    )
