"""The plan tool: the model's half of the plan/approve gate."""

from typing import Any, ClassVar

from tynex.agent.types import AgentTool, ToolContext, ToolResult

PLAN_TOOL_NAME = "submit_plan"


class SubmitPlanTool(AgentTool):
    name = PLAN_TOOL_NAME
    permission_kind = "read"
    label = "plan"
    mutating = False
    description = (
        "Submit a complete implementation plan for the user to approve. Only available in plan mode. "
        "The turn ends when the plan is submitted, so include everything the implementer needs."
    )
    prompt_snippet = "Submit an implementation plan for approval (plan mode only)"
    prompt_guidelines: ClassVar[list[str]] = [
        "In plan mode, investigate first and call submit_plan once the plan is complete.",
        "Do not ask whether to proceed in a message: submit_plan is the approval request.",
    ]
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": (
                    "The plan in Markdown: goal, files and interfaces to change, ordered steps, "
                    "risks, and how the result will be verified."
                ),
            }
        },
        "required": ["plan"],
    }

    def execute(
        self, tool_call_id: str, args: dict[str, Any], signal=None, on_update=None, ctx: ToolContext | None = None
    ) -> ToolResult:
        if not str(args.get("plan", "")).strip():
            raise ValueError("plan must not be empty")
        result = ToolResult.text("Plan submitted. The turn ends here for the user's decision.")
        result.terminate = True  # a planning turn is over once the plan is in
        return result
