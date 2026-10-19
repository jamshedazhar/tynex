"""System prompt construction."""

from tynex.agent.types import AgentTool
from tynex.coding.settings import AGENTS_FILENAME, SYSTEM_FILENAME, find_files, read_user_text

DEFAULT_PREAMBLE = (
    "You are an expert coding assistant operating inside tynex, a coding agent harness. "
    "You help users by reading files, executing commands, editing code, and writing new files."
)


def _build_rules(
    tools: list[AgentTool], read_only_tool_names: set[str], extra_guidelines: list[str] | None = None
) -> str:
    rules: list[str] = []
    seen: set[str] = set()

    def add(rule: str) -> None:
        normalized = rule.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            rules.append(normalized)

    add("Only read or modify files inside the working directory; never access paths outside it")
    names = [t.name for t in tools]
    shell = "bash" if "bash" in names else next((name for name in ("pwsh", "powershell") if name in names), None)
    has_search = any(n in read_only_tool_names for n in names)
    if shell and not has_search:
        examples = "ls, rg, find" if shell == "bash" else "Get-ChildItem, Select-String"
        add(f"Use {shell} for file operations like {examples}")
    for tool in tools:
        for rule in tool.prompt_guidelines:
            add(rule)
    for rule in extra_guidelines or []:
        add(rule)
    add("Be concise in your responses")
    add("Show file paths clearly when working with files")
    return "\n".join(f"- {rule}" for rule in rules)


def build_system_prompt(
    cwd: str,
    tools: list[AgentTool],
    custom_prompt: str | None = None,
    append_system_prompt: str | None = None,
    context_files: list[dict[str, str]] | None = None,
    skills_lines: str | None = None,
    extra_sections: dict[str, str] | None = None,
    extra_guidelines: list[str] | None = None,
) -> tuple[str, dict[str, str]]:
    """Returns (full_prompt_text, sections)."""
    read_only_tool_names = {"grep", "find", "ls"}
    sections: dict[str, str] = {}
    if custom_prompt:
        preamble = custom_prompt
        guidelines = "\n".join(f"- {rule.strip()}" for rule in extra_guidelines or [] if rule.strip())
        if guidelines:
            sections["guidelines"] = guidelines
    else:
        preamble = DEFAULT_PREAMBLE
        visible = [t for t in tools if t.prompt_snippet]
        tool_lines = "\n".join(f"- {t.name}: {t.prompt_snippet}" for t in visible) or "(none)"
        sections["tools"] = (
            f"{tool_lines}\n\nIn addition to the tools above, you may have access to other custom "
            "tools depending on the project."
        )
        sections["rules"] = _build_rules(tools, read_only_tool_names, extra_guidelines)
    if append_system_prompt:
        sections["addendum"] = append_system_prompt
    if context_files:
        rendered = ["Project-specific instructions and guidelines:"]
        for file in context_files:
            rendered.append(f'<project_instructions path="{file["path"]}">\n{file["content"]}\n</project_instructions>')
        sections["project_context"] = "\n\n".join(rendered)
    if skills_lines and tools and any(t.name in ("read", "bash", "pwsh", "powershell") for t in tools):
        sections["skills"] = skills_lines
    sections["cwd"] = cwd.replace("\\", "/")
    for name, content in (extra_sections or {}).items():
        if content:
            sections[name] = content
    parts = [preamble]
    for name, content in sections.items():
        parts.append(f"<{name}>\n{content}\n</{name}>")
    return "\n\n".join(parts), sections


def load_context_files(cwd: str) -> list[dict[str, str]]:
    """Load the nearest AGENTS.md — ``<cwd>/.tynex`` first, then ``<cwd>``, then the built-in one."""
    path = find_files(AGENTS_FILENAME, cwd, one=True)
    content = read_user_text(path) if path else None
    return [{"path": str(path), "content": content}] if content else []


def load_system_md(cwd: str) -> str | None:
    """SYSTEM.md replaces the default system prompt preamble. Project file wins."""
    path = find_files(SYSTEM_FILENAME, cwd, one=True)
    return read_user_text(path) if path is not None else None
