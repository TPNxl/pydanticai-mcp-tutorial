# TODO: so why do we need MCPs?

import logfire
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.mcp import MCPToolset

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


toolset = MCPToolset('http://localhost:23846/mcp')  
# TODO: load model from utils.py
model = 'claude-sonnet-4-6'
agent = Agent(
    model,
    toolsets=[toolset]
)

if __name__ == '__main__':
    message_history: list[ModelMessage] = []

    print("Physics assistant, with access to the MCP server. Make sure 5_mcp_server.py is running. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ('quit', 'exit', 'q'):
            break
        if not user_input:
            continue

        result = agent.run_sync(user_input, message_history=message_history)
        message_history.extend(result.new_messages())
        print(f"Agent: {result.output}\n")
