from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts.chat import ChatPromptTemplate
from agent.tool_function import create_tool_calling_agent
from tools.customer_tools import get_customer_info
from langchain.tools.render import render_text_description
from dotenv import load_dotenv
import param
import os 

load_dotenv()

def read_system_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
class ClaudeAgent(param.Parameterized):
    def __init__(self, tools, **params):
        super(ClaudeAgent, self).__init__(**params)
        self.model = ChatAnthropic(model = "claude-3-haiku-20240307", api_key=os.environ["ANTHROPIC_API_KEY"])
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key='input', output_key="output")
        self.system_prompt = read_system_prompt(f"{os.path.dirname(__file__)}/prompt/system_prompt2.txt")
        self.system_prompt = self.system_prompt.replace("{tools}", render_text_description(tools))
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = create_tool_calling_agent(self.model, tools, self.prompt)
        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory, return_intermediate_steps=False)

    def message(self, query):
        if not query:
            return
        result = self.qa.invoke({"input": query})
        self.answer = result['output']
        return self.answer

    # async def message_conf(self, query):
    #     if not query:
    #         return
        
    #     for step in self.qa.iter({"input": query}):
    #         if output := step.get("intermediate_step"):
    #             action, value = output[0]
    #             if action.tool == "get_customer_info":
    #                 print(f"Ricerco per ID cliente {value}...")
    #                 assert get_customer_info(int(value))
    #             # Ask user if they want to continue
    #             _continue = input("Should the agent continue (Y/n)?:\n") or "Y"
    #             if _continue.lower() != "y":
    #                 break

    async def message_info(self, query):
        if not query:
            return

        async for event in self.qa.astream_events(
            {"input": query},
            version="v1",
        ):
            kind = event["event"]
            if kind == "on_chain_start":
                if (
                    event["name"] == "Agent"
                ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                    print(
                        f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                    )
            elif kind == "on_chain_end":
                if (
                    event["name"] == "Agent"
                ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                    print()
                    print("--")
                    print(
                        f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                    )
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # Empty content in the context of OpenAI means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content
                    print(content, end="|")
            elif kind == "on_tool_start":
                print("--")
                print(
                    f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                )
            elif kind == "on_tool_end":
                print(f"Done tool: {event['name']}")
                print(f"Tool output was: {event['data'].get('output')}")
                print("--")

    def clr_history(self, count=0):
        self.chat_history = []
        return
    


