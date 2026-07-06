# Goal: Call the LLM from the Python file using the library
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'

from pydantic_ai import Agent

from utils.formatting import format_usage
from utils.llm import MODEL, configure_logfire

configure_logfire()

agent = Agent(MODEL)

if __name__ == '__main__':
    result = agent.run_sync('The windy city in the US of A is? (Only 1 sentence)')

    print("--- Agent output ---\n")
    print(result.output) # Should say Chicago, Illinois or something similar.
    print("\n--- Agent usage ---\n")
    print(format_usage(result.usage))
    print()