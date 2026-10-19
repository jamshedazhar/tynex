"""Skills and prompt templates: capability packages and reusable prompts."""

import re
from dataclasses import dataclass
from pathlib import Path

from tynex.coding.settings import find_files, read_user_text

SKILL_FILENAME = "SKILL.md"
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


@dataclass
class Skill:
    name: str
    description: str
    path: str


@dataclass
class PromptTemplate:
    name: str
    content: str
    path: str
    description: str = ""


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    meta: dict[str, str] = {}
    for line in text[3:end].splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip().lower()] = value.strip().strip("'\"")
    return meta, text[end + 4 :].lstrip("\r\n")


def discover_skills(cwd: str) -> list[Skill]:
    """Skills from the project and working dirs first, then the built-in ones; the first definition of a name wins."""
    seen: set[str] = set()
    skills: list[Skill] = []
    for path in find_files(f"skills/*/{SKILL_FILENAME}", cwd):
        text = read_user_text(path)
        if text is None:
            continue
        meta, _ = _split_frontmatter(text)
        name = meta.get("name") or path.parent.name
        if name in seen:
            continue
        seen.add(name)
        skills.append(
            Skill(name=name, description=meta.get("description", "").strip() or f"Skill {name}", path=str(path))
        )
    return skills


def read_skill_instructions(skill: Skill) -> str | None:
    """The SKILL.md body without frontmatter; ``None`` when unreadable or empty."""
    text = read_user_text(Path(skill.path))
    if text is None:
        return None
    return _split_frontmatter(text)[1].strip() or None


def format_skills_for_prompt(skills: list[Skill], read_tool: str) -> str:
    if not skills:
        return ""
    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        f"Read the full skill file with the {read_tool} tool when the task matches its description:",
        "",
    ]
    for skill in skills:
        lines.append(f"- {skill.name}: {skill.description} ({skill.path})")
    return "\n".join(lines)


def discover_prompts(cwd: str) -> dict[str, PromptTemplate]:
    """Prompt templates by name; project files override built-in ones."""
    templates: dict[str, PromptTemplate] = {}
    for file in find_files("prompts/*.md", cwd):
        if file.stem in templates:
            continue  # project templates shadow built-in ones with the same name
        content = read_user_text(file)
        if content is None:
            continue
        templates[file.stem] = PromptTemplate(
            name=file.stem, content=content, path=str(file), description=content.splitlines()[0][:80]
        )
    return templates


def expand_template(template: PromptTemplate, args: str) -> str:
    args = args.strip()
    positional = args.split() if args else []

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "args":
            return args
        if key.isdigit():
            index = int(key) - 1
            return positional[index] if index < len(positional) else ""
        return ""

    expanded = PLACEHOLDER.sub(replace, template.content)
    if (
        args
        and "{{args}}" not in template.content
        and not any(key.isdigit() for key in PLACEHOLDER.findall(template.content))
    ):
        expanded += f"\n\n{args}"
    return expanded.strip()
