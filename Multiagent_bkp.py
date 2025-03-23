import asyncio
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace

weather_agent = Agent(
    name="weather_agent",
    instructions="You answer weather-related queries by responding with 'The weather is 50 degrees.'",
    handoff_description="A weather information agent",
)

politics_agent = Agent(
    name="politics_agent",
    instructions="You answer politics-related queries by responding with 'Hello NewYork.'",
    handoff_description="A politics information agent",
)

manager_agent = Agent(
    name="manager_agent",
    instructions=(
        "You are a smart router agent. "
        "Decide which tool to use based on the user query. "
        "Use 'weather_agent' for weather-related queries. "
        "Use 'politics_agent' for politics-related queries. "
        "Always use the tools. Never answer directly."
    ),
    tools=[
        weather_agent.as_tool(
            tool_name="get_weather",
            tool_description="Provides weather information",
        ),
        politics_agent.as_tool(
            tool_name="get_politics",
            tool_description="Provides politics information",
        ),
    ],
)

# Optional: Synthesizer Agent for post-processing (if needed)
synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="You review responses from the tools and finalize the answer.",
)

async def main():
    msg = input("Ask me about weather or politics: ")

    with trace("Manager Orchestrator Run"):
        manager_result = await Runner.run(manager_agent, msg)

        for item in manager_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Agent Response: {text}")
        synthesizer_result = await Runner.run(
            synthesizer_agent, manager_result.to_input_list()
        )

    print(f"\n\n Final response:\n{synthesizer_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
