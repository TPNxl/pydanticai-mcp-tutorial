# Goal: How to stack multiple messages
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'
#
# Version 1: each call is independent — the agent has no memory of prior messages.
# Version 2: message history is passed between calls, giving the agent full context.
#
# Usage: python 2_context_multiple_calls.py [1|2]  (default: 2)

import sys

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from utils import MODEL, pretty_print_history

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent(MODEL, instructions=(
    "Answer with at most 1 sentence."
))

@agent.system_prompt
def system_prompt():
    return "The secret letter is R."

# The differences:
# Version 1: each call is independent — the agent has no memory of prior messages.
# Version 2: message history is passed between calls, giving the agent full context.
# However, both agents remember the system prompt, which is included in every call regardless of message history.

def run_version_1():
    # Without message history: the second call has no memory of the first question.
    result = agent.run_sync('What\'s 1+1?')
    print(result.output)
    result = agent.run_sync('What is the secret letter?')
    print(result.output)
    result = agent.run_sync('What question did I ask at the start?')
    print(result.output)


def run_version_2():
    message_history: list[ModelMessage] = []

    def run_and_update(message: str):
        nonlocal message_history
        result = agent.run_sync(message, message_history=message_history)
        print(result.output)
        message_history.extend(result.new_messages())

    run_and_update('What\'s 1+1?')
    run_and_update('What is the secret letter?')
    run_and_update('What question did I ask at the start?')

    print("\n--- Message history ---\n")
    pretty_print_history(message_history)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Usage: python 2_context_multiple_calls.py [1|2]")
        sys.exit(1)
    version = sys.argv[1]
    if version == '1':
        print("Running version 1 (no message history):\n")
        run_version_1()
    else:
        print("Running version 2 (with message history):\n")
        run_version_2()
