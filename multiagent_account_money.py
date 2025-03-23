import asyncio
import speech_recognition as sr
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace

acat_agent = Agent(
    name="acat_agent",
    instructions=(
        "You answer account transfer related queries by responding with 'Your account transfer is in process.' \n"
        "This is a message from ACAT agent.'"
    ),
    handoff_description="Handles account transfer information.",
)

moneymovement_agent = Agent(
    name="moneymovement_agent",
    instructions=(
        "You answer money transfer queries by responding with 'Your money transfer is in process.' \n"
        "This is a message from Money Movement agent.'"
    ),
    handoff_description="Handles money movement information.",
)

manager_agent = Agent(
    name="manager_agent",
    instructions=(
        "You are a router agent. "
        "Route the query to the appropriate tool based on the user's intent. "
        "Use 'acat_agent' for account transfer queries. "
        "Use 'moneymovement_agent' for money transfer queries. "
    ),
    tools=[
        acat_agent.as_tool(
            tool_name="get_acat_info",
            tool_description="Provides account transfer information."
        ),
        moneymovement_agent.as_tool(
            tool_name="get_moneymovement_info",
            tool_description="Provides money movement information."
        ),
    ],
)

synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="Review the responses from the tools and finalize the answer.",
)

def get_speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n Speak now... (or type if you prefer)")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f" You said: {text}")
        return text
    except sr.UnknownValueError:
        print(" Sorry, could not understand the audio.")
        return None
    except sr.RequestError:
        print(" Speech Recognition service is unavailable.")
        return None

async def main():
    while True:
        choice = input("\nDo you want to (1) Speak or (2) Type? Enter 1 or 2 (or 'exit' to quit): ").strip()
        if choice.lower() == "exit":
            print("Exiting the conversation. Bye!")
            break
        if choice == "1":
            user_query = get_speech_input()
            if not user_query:
                continue  
        elif choice == "2":
            user_query = input("Type your query: ").strip()
        else:
            print("Invalid choice. Please enter 1, 2, or 'exit'.")
            continue
        if user_query.lower() == "exit":
            print("Exiting the conversation. Bye!")
            break

        with trace("Manager Orchestrator Run"):
            manager_result = await Runner.run(manager_agent, user_query)
            for item in manager_result.new_items:
                if isinstance(item, MessageOutputItem):
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        print(f"\n Agent Response:\n{text}")

            synthesizer_result = await Runner.run(
                synthesizer_agent, manager_result.to_input_list()
            )

        print(f"\n Final Response:\n{synthesizer_result.final_output}")
        
if __name__ == "__main__":
    asyncio.run(main())
