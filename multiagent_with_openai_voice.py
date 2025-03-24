import asyncio
import random
from agents import Agent, function_tool
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoicePipeline,
)
from util import AudioPlayer, record_audio


@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    print(f"[debug] get_weather called with city: {city}")
    return "The weather is 50 degrees."


@function_tool
def get_politics(query: str) -> str:
    """Handle politics related queries."""
    print(f"[debug] get_politics called with query: {query}")
    return "Hello NewYork."


# Define agents
weather_agent = Agent(
    name="WeatherAgent",
    handoff_description="Handles weather-related queries.",
    instructions="You provide weather updates.",
    model="gpt-4o-mini",
    tools=[get_weather],
)

politics_agent = Agent(
    name="PoliticsAgent",
    handoff_description="Handles politics-related queries.",
    instructions="You answer politics queries politely.",
    model="gpt-4o-mini",
    tools=[get_politics],
)

# Manager agent acts as router
manager_agent = Agent(
    name="ManagerAgent",
    instructions=prompt_with_handoff_instructions(
        "Route weather-related queries to the weather agent and politics queries to the politics agent. Never answer directly."
    ),
    model="gpt-4o-mini",
    handoffs=[weather_agent, politics_agent],
    tools=[get_weather, get_politics],
)


class WorkflowCallbacks(SingleAgentWorkflowCallbacks):
    def on_run(self, workflow: SingleAgentVoiceWorkflow, transcription: str) -> None:
        print(f"[debug] on_run called with transcription: {transcription}")


synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="You review responses from the tools and finalize the answer.",
)
async def voice_flow():
    pipeline = VoicePipeline(
        workflow=SingleAgentVoiceWorkflow(manager_agent, callbacks=WorkflowCallbacks())
    )

    audio_input = AudioInput(buffer=record_audio())

    result = await pipeline.run(audio_input)

    with AudioPlayer() as player:
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                player.add_audio(event.data)
                print("Received audio")
            elif event.type == "voice_stream_event_lifecycle":
                print(f"Received lifecycle event: {event.event}")


async def text_flow():
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
async def main():
    mode = input("Choose mode (voice/text): ").strip().lower()

    if mode == "voice":
        await voice_flow()
    elif mode == "text":
        await text_flow()
    else:
        print("Invalid option. Please choose 'voice' or 'text'.")


if __name__ == "__main__":
    asyncio.run(main())
