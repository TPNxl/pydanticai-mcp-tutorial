# Goal: How to stack multiple messages
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'
#
# Notice: each agent has its own system prompt and does not see the other agent's.
# Despite being the same model, they argue from different perspectives based on their instructions.

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart

from utils import MODEL, pretty_print_history

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent_A = Agent(MODEL, name="Agent A")
agent_B = Agent(MODEL, name="Agent B")


def make_debate_prompt(food: str) -> str:
    return (
        f"You are in a debate with another agent about which food is better. "
        f"Your favorite food is {food}. "
        f"You must argue that {food} is better than the other agent's food, "
        f"1 sentence at a time. "
        f"Do not include your name at the start of the response, as it is formatted automatically."
    )


@agent_A.system_prompt
def system_prompt_A():
    return make_debate_prompt("pizza")


@agent_B.system_prompt
def system_prompt_B():
    return make_debate_prompt("sushi")


message_history: list[ModelMessage] = []


def run_agent(agent: Agent, first_run: bool = False):
    result = agent.run_sync(
        user_prompt="start the argument." if first_run else "continue the argument.",
        message_history=message_history,
    )

    for msg in result.new_messages():
        if isinstance(msg, ModelRequest):
            # Strip system prompts (agent-private); keep user prompts for conversation context
            msg.parts = [p for p in msg.parts if not isinstance(p, SystemPromptPart)]
            if msg.parts:
                message_history.append(msg)
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart) and not part.content.startswith(f"{agent.name}:"):
                    part.content = f"{agent.name}: {part.content}"
            message_history.append(msg)

    return result.usage


if __name__ == '__main__':
    usages = []
    for i in range(5):
        usages.append(run_agent(agent_A, first_run=(i == 0)))
        usages.append(run_agent(agent_B))

    print("--- Debate transcript ---\n")
    pretty_print_history(message_history, show_user_prompts=False, label_agents=False)

    total_input = sum(u.input_tokens or 0 for u in usages)
    total_output = sum(u.output_tokens or 0 for u in usages)
    total_requests = sum(u.requests or 0 for u in usages)

    print("\n--- Total usage ---")
    print(f"  Requests:      {total_requests}")
    print(f"  Input tokens:  {total_input}")
    print(f"  Output tokens: {total_output}")
    print(f"  Total tokens:  {total_input + total_output}")
