# Goal: How to stack multiple messages
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'
#
# Version 1: each call is independent — the agent has no memory of prior messages.
# Version 2: message history is passed between calls, giving the agent full context.
#
# Usage: python 2_context_multiple_calls.py [1|2]  (default: 2)

import sys

from pydantic_ai import Agent, AgentRunResult

from utils.formatting import pretty_print_history
from utils.llm import MODEL, configure_logfire

configure_logfire()

agent = Agent(MODEL, instructions=(
    "Answer with at most 1 sentence."
))

@agent.system_prompt
def system_prompt():
    return "The secret letter is R."

# The differences:
# Version 1: each call is independent — the agent has no memory of prior messages.
# Version 2: message history is passed between calls, giving the agent full context.

# However, both agents remember the system prompt.
# The system prompt is passed in when message_history in run_sync(...) is not populated, and is assumed to be in the history otherwise.
# Notice why the version 2 agent remembers the system prompt:
# the first run_sync() populates it and the all_messages() in the further run_sync()s carry it forward.

def run_version_1():
    # Without message history: the second call has no memory of the first question.
    result = agent.run_sync('What\'s 1+1?')
    print(result.output)
    result = agent.run_sync('What is the secret letter?')
    print(result.output)
    result = agent.run_sync('What question did I ask at the start?')
    print(result.output)

    # No common message history to print here

def run_version_2():
    result = None
    def run_and_update(message: str, result: AgentRunResult | None):
        if result:
            result = agent.run_sync(message, message_history=result.all_messages())
        else:
            result = agent.run_sync(message)
        print(result.output)
        return result

    result = run_and_update('What\'s 1+1?', None)
    result = run_and_update('What is the secret letter?', result)
    result = run_and_update('What question did I ask at the start?', result)

    print("\n--- Message history ---\n")
    pretty_print_history(result.all_messages())


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
