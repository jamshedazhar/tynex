"""Arrow-key list selector shared by the interactive TUI and the CLI login flow."""

from typing import Any


def select_option(
    title: str, options: list[str], *, input: Any = None, output: Any = None, selected: int = 0
) -> int | None:
    """Arrow-key list selector."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    if not options:
        return None
    state = {"index": min(max(selected, 0), len(options) - 1)}

    def render() -> FormattedText:
        fragments: list[tuple[str, str]] = [("class:title", f"{title}\n")]
        for index, option in enumerate(options):
            is_selected = index == state["index"]
            marker = "❯ " if is_selected else "  "  # noqa: RUF001 — intentional prompt glyph
            fragments.append(("class:selected" if is_selected else "", f"{marker}{option}\n"))
        fragments.append(("class:hint", "(↑/↓ to move, Enter to select, Esc to cancel)"))
        return FormattedText(fragments)

    keys = KeyBindings()

    @keys.add("up")
    @keys.add("k")
    def _move_up(event: Any) -> None:
        state["index"] = (state["index"] - 1) % len(options)

    @keys.add("down")
    @keys.add("j")
    def _move_down(event: Any) -> None:
        state["index"] = (state["index"] + 1) % len(options)

    @keys.add("enter")
    def _accept(event: Any) -> None:
        event.app.exit(result=state["index"])

    @keys.add("escape")
    @keys.add("c-c")
    def _cancel(event: Any) -> None:
        event.app.exit(result=None)

    style = Style.from_dict({"title": "bold", "selected": "ansicyan bold", "hint": "italic #888888"})
    app: Application[int | None] = Application(
        layout=Layout(Window(FormattedTextControl(render), dont_extend_height=True)),
        key_bindings=keys,
        style=style,
        input=input,
        output=output,
    )
    return app.run()
