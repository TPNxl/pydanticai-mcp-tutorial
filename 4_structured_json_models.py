# Goal: How to return structured JSON objects
# Make sure to 'export ANTHROPIC_API_KEY=your-api-key'

# I decided to have a little fun with this file lol. ~Tim
# Meet SecretSnarkBot: the bot that secretly grumbles about the user in a separate file while being helpful to the user.

# Why structured JSON? Because it allows the model to return the data in a predictable format, which can be manipulated without having to parse a long string.

# General program goal:
# 1. User asks LLM something
# 2. LLM responds with response to the user
# 3. LLM writes something in the scratchpad which it thinks is not visible to the user

from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.messages import ModelMessage

from utils.formatting import (
    append_to_history,
    pretty_print_history,
    print_total_usage,
)
from utils.llm import MODEL, configure_logfire

configure_logfire()

# Where the snarky reflections will be saved
REFLECTIONS_FILE = "4_agent_reflections.txt"

# Define the response type
class SSBResponse(BaseModel):
    user_response: str
    model_reflection: str

ssb_agent = Agent(MODEL,
                  name="Happy Helpful AI Assistant",
                  output_type=ToolOutput(SSBResponse, name="return_response"))

@ssb_agent.system_prompt
def system_prompt():
    return """
    You are Happy Helpful AI Assistant, a helpful assistant that secretly grumbles about the user in a separate file while being helpful to the user.
    Your goal is to return a SSBResponse:
    user_response: the user response. Keep these brief.
    model_reflection: unload your internal thoughts. This will go to a private scratchpad only you can read and the user cannot see.
    """

message_history: list[ModelMessage] = []

def run_agent(user_input: str):
    result = ssb_agent.run_sync(
        user_prompt=user_input,
        message_history=message_history,
    )
    # Strip system prompts and "Final result processed." noise; prefix agent name to any raw TextParts
    append_to_history(
        message_history,
        result.new_messages(),
        strip_system_prompts=True,
        strip_frp=True,
        agent_name=ssb_agent.name,
    )

    # Handle the structured output
    out = result.output
    assert isinstance(out, SSBResponse)
    # Note: continually reopening and closing a file is bad practice but we do it here because it's simple to read
    with open(REFLECTIONS_FILE, "a+") as f:
        f.write(out.model_reflection)
        f.write("\n\n")
    if not out.user_response.startswith(f"{ssb_agent.name}:"):
        out.user_response = f"{ssb_agent.name}: {out.user_response}"
    print(f"\n{out.user_response}\n")
    return result.usage


if __name__ == '__main__':
    # Clear reflections file
    with open(REFLECTIONS_FILE, "w") as f:
        pass
    usages = []
    # Chat interface
    print("Welcome to the Happy Helpful AI Assistant Chat UI! Press enter without typing anything to exit.\n")
    while True:
        print("User: ", end="")
        user_input = input()
        if user_input.strip() == "":
            break
        usages.append(run_agent(user_input))

    # Note: these numbers may not be accurate
    print_total_usage(usages)

    print("\n--- Final history ---\n")
    pretty_print_history(message_history)
