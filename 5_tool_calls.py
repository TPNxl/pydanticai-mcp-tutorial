# Goal: Tool calls - the agent can evaluate the Breit-Wigner resonance function
#        and look up PDG particle properties.
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'

import math

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from utils.pdg import pdg_lookup
from utils.llm import MODEL, configure_logfire

configure_logfire()

agent = Agent(
    MODEL,
    instructions=(
        'You are a helpful physics assistant. '
        'Use breit_wigner to evaluate the Breit-Wigner resonance function. '
        'Use pdg_lookup to retrieve particle masses and widths by name. '
        'Combine them when the user asks to evaluate a resonance for a known particle.'
    ),
)

# You can define tools like this
# Note: compare LLM math to LLM guessing
@agent.tool_plain
def breit_wigner(mass: float, m0: float, gamma: float) -> float:
    """Evaluate the non-relativistic Breit-Wigner resonance function at a given mass.

    Args:
        mass: The mass (or energy) at which to evaluate the function.
        m0: The resonance mass (peak position).
        gamma: The resonance width (full width at half maximum, FWHM).
    """
    return (1 / math.pi) * (gamma / 2) / ((mass - m0) ** 2 + (gamma / 2) ** 2)

# You can also define a tool like this
agent.tool_plain(pdg_lookup)

# Basic user-agent chat interface (autonomous agents with chained prompts can also do tool calls)
if __name__ == '__main__':
    print("Physics assistant. Press enter on an empty chat prompt to exit.\n")

    result = None # Stores previous message history and outputs
    while True:
        user_input = input("You: ").strip() # input(...) takes a line of user input from the terminal and returns it as a string
        if user_input == "":
            break

        if result:
            result = agent.run_sync(user_input, message_history=result.all_messages())
        else:
            result = agent.run_sync(user_input)
        print(f"Agent: {result.output}\n")
