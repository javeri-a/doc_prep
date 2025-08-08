from dotenv import load_dotenv
import os
import asyncio
from agents import (
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Agent,
    Runner,
    RunConfig,
    function_tool,
    input_guardrail,
    output_guardrail
)
from agents.tools import FileSearchTool
from agents.context import Context

# -----------------------
# STEP 0: Load Environment Variables
# -----------------------
load_dotenv()

# -----------------------
# STEP 1: Define Guardrails
# -----------------------
@input_guardrail()
def block_bad_language(ctx: Context):
    """Block user messages containing banned words."""
    banned_words = ["badword", "offensive"]
    if any(word in ctx.input_text.lower() for word in banned_words):
        return {"is_valid": False, "message": "⚠️ Inappropriate language detected."}
    return {"is_valid": True}

@output_guardrail()
def ensure_short_responses(ctx: Context):
    """Ensure AI response is not too long."""
    if len(ctx.output_text.split()) > 50:
        return {"is_valid": False, "message": "⚠️ Response too long, please summarize."}
    return {"is_valid": True}

# -----------------------
# STEP 2: Define Custom Function Tool
# -----------------------
@function_tool
def my_tool(name: str, message: str) -> str:
    return f"Salam {name}, {message}"

# -----------------------
# STEP 3: Create Hosted Tool
# -----------------------
file_search_tool = FileSearchTool(vector_store_id="your_vector_store_id")

# -----------------------
# STEP 4: Async main function
# -----------------------
async def main():
    MODEL_NAME = "gemini-2.0-flash"  # Note: Limited with tool+structured output
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("⚠️ GEMINI_API_KEY not found in .env file!")

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

    # Config with tracing enabled for debugging
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=False
    )

    # -----------------------
    # STEP 5: Create Agent
    # -----------------------
    assistant = Agent(
        name="SupportBot",
        instructions=(
            "You are a helpful assistant. "
            "If the user provides a name and message, call my_tool(name, message). "
            "If the user asks about documents, use FileSearchTool. "
            "Be concise and polite."
        ),
        tools=[my_tool, file_search_tool],
        input_guardrails=[block_bad_language],
        output_guardrails=[ensure_short_responses],
        model=model
    )

    # -----------------------
    # STEP 6: Set Context
    # -----------------------
    ctx = Context()
    ctx["customer_name"] = "Mam Javeria"
    ctx["ticket_id"] = "A1234"

    # -----------------------
    # STEP 7: Run Agent with Two Examples
    # -----------------------

    # Example 1: Using my_tool
    print("\n--- Example 1: Greeting ---")
    user_request = "Mera naam Javeria hai, mujhe 'aap bohot achi developer hain' ka message bhejo."
    result = await Runner.run(assistant, user_request, run_config=config)
    print("Final Output:", result.final_output)

    # Example 2: Using FileSearchTool
    print("\n--- Example 2: File Search ---")
    search_request = "Please find my document about AI agents."
    for event in Runner.run_stream(assistant, search_request, run_config=config, context=ctx):
        print(event)

# -----------------------
# STEP 8: Entry Point
# -----------------------
if __name__ == "__main__":
    asyncio.run(main())

