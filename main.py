from tools.customer_tools import get_customer_info, get_customer_bills, sign_contract
from tools.retriver_tools import get_all_info
from agent.claude_agent import ClaudeAgent
from agent.openai_agent import OpenAiAgent
from agent.ollama_agent import OllamaAgent
import asyncio

async def main():
    try:
        tools = [get_customer_info, get_customer_bills, sign_contract, get_all_info]
        agent = OpenAiAgent(tools)
        print("Benvenuto! Come posso aiutarti?")
        while True:
            user_input = input("> ")
            if user_input.lower() == "q":
                print("Arrivederci!")
                break
            else:
                #await agent.message_conf(user_input)
                #await agent.message_stream(user_input)
                response = agent.message(user_input)
                print(response)
    finally:
        agent.clr_history()

if __name__ == "__main__":
    asyncio.run(main())
