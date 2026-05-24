# Goal: Tool calls - the agent can evaluate the Breit-Wigner resonance function
#        and look up PDG particle properties.
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'

import math

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from pdg import pdg_lookup
from utils import MODEL

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

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


if __name__ == '__main__':
    message_history: list[ModelMessage] = []

    print("Physics assistant. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue

        result = agent.run_sync(user_input, message_history=message_history)
        message_history.extend(result.new_messages())
        print(f"Agent: {result.output}\n")
