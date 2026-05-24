# Goal: Call the LLM from the Python file using the library
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'

import logfire
from pydantic_ai import Agent

from utils import MODEL, format_usage

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent(MODEL)


if __name__ == '__main__':
    result = agent.run_sync('The windy city in the US of A is? (Only 1 sentence)')

    print("--- Agent output ---\n")
    print(result.output)
    print("\n--- Agent usage ---\n")
    print(format_usage(result.usage))
    print()
