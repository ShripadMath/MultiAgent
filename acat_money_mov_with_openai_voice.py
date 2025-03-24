import asyncio
import random
from agents import Agent, function_tool
from agents import ItemHelpers, MessageOutputItem, Runner, trace
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    SingleAgentWorkflowCallbacks,
    VoicePipeline,
)
from util import AudioPlayer, record_audio
import re


# Tool to get account details (ACAT)
@function_tool
def get_acat_details(account_number: str) -> str:
    """Handle account transfer process."""
    print(f"[debug] get_acat_details called with account_number: {account_number}")
    return f"Your account transfer with account number {account_number} is in process."

@function_tool
def get_money_movement_status(account_number: str, transfer_date: str) -> str:
    """Handle money movement status check."""
    print(f"[debug] get_money_movement_status called with account_number: {account_number} and transfer_date: {transfer_date}")
    return f"Money transfer done on {transfer_date} for account number {account_number} is in progress."


# Define agents
account_agent = Agent(
    name="AccountAgent",
    handoff_description="Handles account transfer and account-related queries.",
    instructions="Ask for account number if not provided and then proceed with account details.",
    model="gpt-4o-mini",
    tools=[get_acat_details],
)

money_movement_agent = Agent(
    name="MoneyMovementAgent",
    handoff_description="Handles money transfer and movement queries.",
    instructions="Ask for account number and transfer date if not provided. Then proceed with money movement status.",
    model="gpt-4o-mini",
    tools=[get_money_movement_status],
)
# Manager agent acts as router
manager_agent = Agent(
    name="ManagerAgent",
    instructions=prompt_with_handoff_instructions(
        """"
        Route account-related queries to the AccountAgent when only the account number is provided. 
        Route money movement queries to the MoneyMovementAgent when both the account number and the transfer date are provided. 
        Never answer directly.
        """
    ),
    model="gpt-4o-mini",
    handoffs=[account_agent, money_movement_agent],
    tools=[get_acat_details, get_money_movement_status],
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


def extract_account_number(msg):
    """
    Extracts a 6 to 12 digit account number from the message.
    """
    match = re.search(r'\b\d{6,12}\b', msg)
    return match.group() if match else None


async def text_flow():
    choice = input("Please select an option: (1) Account Transfer (2) Money Transfer: ").strip()
    if choice == '1':
        print("You selected Account Transfer. Please provide your account number.")
        while True:
            msg = input("Enter account number: ").strip()
            account_number = extract_account_number(msg)
            if account_number:
                print(f'Extracted Account: {account_number}')
                break
            else:
                print("Invalid account number format. Please try again.")
        with trace("Manager Orchestrator Run - Account Transfer"):
            manager_result = await Runner.run(manager_agent, account_number)

    elif choice == '2':
        print("You selected Money Transfer. Please provide your account number and transfer date.")
        while True:
            msg = input("Enter account number: ").strip()
            account_number = extract_account_number(msg)
            if account_number:
                print(f'Extracted Account: {account_number}')
                break
            else:
                print("Invalid account number format. Please try again.")
        while True:
            msg = input("Enter the transfer date (in format dd-mm-yyyy or dd/mm/yyyy): ").strip()
            match = re.search(r'\b\d{2}[-/]\d{2}[-/]\d{4}\b', msg)
            if match:
                transfer_date = match.group()
                print(f'Extracted Transfer Date: {transfer_date}')
                break
            else:
                print("Invalid date format. Please try again.")
        final_input = f"Account number: {account_number}, Transfer date: {transfer_date}"
        with trace("Manager Orchestrator Run - Money Transfer"):
            manager_result = await Runner.run(manager_agent, final_input)

    else:
        print("Invalid choice. Please select '1' for Account Transfer or '2' for Money Transfer.")
        return
    for item in manager_result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text:
                print(f"  - Agent Response: {text}")

    synthesizer_result = await Runner.run(synthesizer_agent, manager_result.to_input_list())
    print(f"\n\nFinal response:\n{synthesizer_result.final_output}")

async def main():
    while True:
        mode = input("Choose mode (voice/text) or type 'exit' to quit: ").strip().lower()

        if mode == "voice":
            await voice_flow()
        elif mode == "text":
            await text_flow()
        elif mode == "exit":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid option. Please choose 'voice', 'text', or 'exit'.")


if __name__ == "__main__":
    asyncio.run(main())