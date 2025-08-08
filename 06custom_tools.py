# from agents import Agent, Runner, function_tool

# @function_tool
# def my_tool (name: str, message: str)-> str:
#        return f"Salam {name}, {message}"

# agent = Agent(
#     name="GreetingAgent",
#     instructions="User ka naam aur message le kar mera_tool use karo.",
#     tools=[my_tool]
# )
# if __name__ == "__main__":
#     user_request = "Mera naam Javeria hai, mujhe 'aap bohot achi developer hain' ka message bhejo."
#     result = Runner.run_sync(agent, user_request)
#     print(result.final_output)




from dotenv import load_dotenv
import os
import asyncio
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner, RunConfig, function_tool

# Load environment variables
load_dotenv()

# 1) Custom function tool
@function_tool
def my_tool(name: str, message: str) -> str:
    return f"Salam {name}, {message}"

# 2) Async main function
async def main():
    MODEL_NAME = "gemini-2.0-flash"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # External Gemini client
    external_client = AsyncOpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    # Model setup
    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # 3) Agent with tool
    assistant = Agent(
        name="GreetingAgent",
        instructions="User ka naam aur message le kar my_tool use karo.",
        tools=[my_tool],
        model=model
    )

    # 4) Run the agent
    user_request = "Mera naam Javeria hai, mujhe 'aap bohot achi developer hain' ka message bhejo."
    result = await Runner.run(assistant, user_request, run_config=config)
    print(result.final_output)

# Entry point
if __name__ == "__main__":
    asyncio.run(main())
