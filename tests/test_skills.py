"""Built-in (packaged) skill discovery and user-skill precedence."""

from pathlib import Path

import tynex
from tynex.coding.skills import discover_skills


def _write_skill(tmp_path: Path, name: str, description: str) -> Path:
    skill_dir = tmp_path / ".tynex" / "skills" / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n", encoding="utf-8")
    return skill_file


def test_discovers_builtin_code_review_skill(tmp_path):
    found = discover_skills(str(tmp_path))
    builtin = next((s for s in found if s.name == "code-review"), None)
    assert builtin is not None
    assert builtin.path == str(Path(tynex.__file__).resolve().parent / "agent" / "skills" / "code-review" / "SKILL.md")
    assert "static analysis" in builtin.description


def test_project_skill_shadows_builtin(tmp_path):
    path = _write_skill(tmp_path, "code-review", "Project override")
    found = discover_skills(str(tmp_path))
    overrides = [s for s in found if s.name == "code-review"]
    assert len(overrides) == 1
    assert overrides[0].path == str(path)
    assert overrides[0].description == "Project override"


def test_builtin_skills_can_be_absent(builtin_root, tmp_path):
    assert discover_skills(str(tmp_path)) == []
