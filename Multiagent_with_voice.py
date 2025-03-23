import asyncio
import speech_recognition as sr
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace



weather_agent = Agent(
    name="acat_agent",
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


synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="You review responses from the tools and finalize the answer.",
)


def listen_to_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source) 
        try:
            print("Recognizing speech...")
            text = recognizer.recognize_google(audio)
            print(f"Recognized text: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return ""
        except sr.RequestError:
            print("Sorry, the speech service is down.")
            return ""

async def main():
    while True:
        choice = input("Do you want to (1) speak or (2) type your query? (Type 'exit' to end): ").strip().lower()
        if choice == "1":
            msg = listen_to_speech()
        elif choice == "2":
            msg = input("Type your query: ").strip()
        elif choice == "exit":
            print("Exiting communication...")
            break
        else:
            print("Invalid choice. Please enter '1' to speak, '2' to type, or 'exit' to end.")
            continue

        if not msg:
            continue
        print(f"You said: {msg}")

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
