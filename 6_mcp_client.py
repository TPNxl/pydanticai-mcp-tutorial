# TODO: so why do we need MCPs?

from pydantic_ai import Agent, ModelMessage
from pydantic_ai.mcp import MCPToolset

from utils.llm import MODEL, configure_logfire

configure_logfire()

toolset = MCPToolset('http://localhost:23846/mcp')  
agent = Agent(
    MODEL,
    toolsets=[toolset]
)

if __name__ == '__main__':
    print("Physics assistant, with access to the MCP server. Make sure 6_mcp_server.py is running. Press enter on an empty prompt to exit.\n")

    result = None

    while True:
        user_input = input("You: ").strip()
        if user_input == "":
            break

        if result:
            result = agent.run_sync(user_input, message_history=result.all_messages())
        else:
            result = agent.run_sync(user_input)
        print(f"Agent: {result.output}\n")
