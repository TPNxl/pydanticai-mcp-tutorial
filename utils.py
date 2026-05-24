from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart

MODEL = 'claude-sonnet-4-6'


def format_usage(usage) -> str:
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
                    print(f"User:  {part.content}")
        elif isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart):
                    print(f"Agent: {part.content}" if label_agents else part.content)
