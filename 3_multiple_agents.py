# Goal: How to stack multiple messages
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'
#
# Notice: each agent has its own system prompt and does not see the other agent's.
# Despite being the same model, they argue from different perspectives based on their instructions.

from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.messages import ModelMessage

from utils.formatting import append_to_history, pretty_print_history, print_total_usage
from utils.llm import MODEL, configure_logfire

configure_logfire()

# Note: we use ReinjectSystemPrompt to ensure the agent always has a system prompt even when we remove it from the shared history
agent_A = Agent(MODEL, name="Agent A", capabilities=[ReinjectSystemPrompt()])
agent_B = Agent(MODEL, name="Agent B", capabilities=[ReinjectSystemPrompt()])

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
        user_prompt="start the argument." if first_run else "continue the argument, addressing the other agent's concerns.",
        message_history=message_history,
    )
    # Note: we use a helper method here to ensure formatting is done correctly in the chat between multiple agents
    # Alternatively a structured response object could also solve this by ensuring adherence
    append_to_history(
        message_history,
        result.new_messages(),
        strip_system_prompts=True,
        agent_name=agent.name,
    )
    return result.usage


if __name__ == '__main__':
    usages = []
    for i in range(5):
        usages.append(run_agent(agent_A, first_run=(i == 0)))
        usages.append(run_agent(agent_B))

    print("--- Debate transcript ---\n")
    pretty_print_history(message_history, show_user_prompts=False, label_agents=False)

    print_total_usage(usages)
