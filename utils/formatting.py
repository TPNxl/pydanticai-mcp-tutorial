from pydantic_ai import ModelRequestPart, ToolCallPart, ToolReturnPart
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

# ---------------------------------------------------------------------------
# Usage helpers
# ---------------------------------------------------------------------------

def format_usage(usage) -> str:
    """Format a single Usage object (returned by result.usage) as a string."""
    total = (usage.input_tokens or 0) + (usage.output_tokens or 0)
    lines = [
        f"  Requests:      {usage.requests}",
        f"  Input tokens:  {usage.input_tokens}",
        f"  Output tokens: {usage.output_tokens}",
        f"  Total tokens:  {total}",
    ]
    if usage.details:
        cache_created = usage.details.get('cache_creation_input_tokens', 0)
        cache_read = usage.details.get('cache_read_input_tokens', 0)
        if cache_created or cache_read:
            lines += [
                f"  Cache created: {cache_created}",
                f"  Cache read:    {cache_read}",
            ]
    return "\n".join(lines)


def sum_and_format_usages(usages: list) -> str:
    """Sum a list of Usage objects and return a formatted string."""
    total_input = sum(u.input_tokens or 0 for u in usages)
    total_output = sum(u.output_tokens or 0 for u in usages)
    total_requests = sum(u.requests or 0 for u in usages)
    return "\n".join([
        f"  Requests:      {total_requests}",
        f"  Input tokens:  {total_input}",
        f"  Output tokens: {total_output}",
        f"  Total tokens:  {total_input + total_output}",
    ])


def print_total_usage(usages: list) -> None:
    """Print a formatted total-usage block from a list of Usage objects."""
    print("\n--- Total usage ---")
    print(sum_and_format_usages(usages))


def count_and_format_usage(history: list[ModelMessage]) -> str:
    """Sum token usage across all LLM responses in a message history.

    Each ModelResponse carries its own usage, including intermediate calls
    made during tool use, so this captures the full cost of a conversation.
    """
    total_input = total_output = total_requests = 0
    for msg in history:
        if isinstance(msg, ModelResponse):
            u = msg.usage
            total_input += u.input_tokens or 0
            total_output += u.output_tokens or 0
            total_requests += u.requests or 0
    total = total_input + total_output
    return "\n".join([
        f"  Requests:      {total_requests}",
        f"  Input tokens:  {total_input}",
        f"  Output tokens: {total_output}",
        f"  Total tokens:  {total}",
    ])


# ---------------------------------------------------------------------------
# Message-history helpers
# ---------------------------------------------------------------------------

def is_frp_msg(p: ModelRequestPart) -> bool:
    """Return True for the 'Final result processed.' sentinel ToolReturnPart."""
    return (
        isinstance(p, ToolReturnPart)
        and p.tool_name == "return_response"
        and p.content == "Final result processed."
    )


def append_to_history(
    history: list[ModelMessage],
    new_messages: list[ModelMessage],
    *,
    strip_system_prompts: bool = True,
    strip_frp: bool = False,
    agent_name: str | None = None,
) -> None:
    """Filter new_messages and append them to the shared history.

    Args:
        history:              The shared conversation history to extend.
        new_messages:         Messages from ``result.new_messages()``.
        strip_system_prompts: Remove SystemPromptPart entries so each agent's
                              private instructions stay out of the shared log.
        strip_frp:            Remove "Final result processed." ToolReturnParts
                              produced by structured-output tools.
        agent_name:           When provided, prefix every TextPart in a
                              ModelResponse with "<agent_name>: " (idempotent).
    """
    for msg in new_messages:
        if isinstance(msg, ModelRequest):
            parts = list(msg.parts)
            if strip_system_prompts:
                parts = [p for p in parts if not isinstance(p, SystemPromptPart)]
            if strip_frp:
                parts = [p for p in parts if not is_frp_msg(p)]
            msg.parts = parts
            if not parts:
                continue  # skip empty requests (nothing useful to keep)
            history.append(msg)

        elif isinstance(msg, ModelResponse):
            if agent_name:
                for part in msg.parts:
                    if (
                        isinstance(part, TextPart)
                        and not part.content.startswith(f"{agent_name}:")
                    ):
                        part.content = f"{agent_name}: {part.content}"
            history.append(msg)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------

def pretty_print_history(
    history: list[ModelMessage],
    *,
    show_user_prompts: bool = True,
    label_agents: bool = True,
) -> None:
    for msg in history:
        if show_user_prompts and isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    print(f"User: {part.content}")
                elif is_frp_msg(part):
                    print(f"<Final result processed ToolReturnPart>")
                else:
                    print(f"Other type of part: {part}")
                print()
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    print(f"Agent: {part.content}" if label_agents else part.content)
                elif isinstance(part, ToolCallPart):
                    print(f"Agent tool call:\n\tTool name: {part.tool_name}\n\tTool args: {part.args}")
                else:
                    print(f"Agent other part: {part}")
                print()
