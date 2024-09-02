from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import ChatPromptTemplate
from typing import Any, Dict
from langchain_experimental.llms.ollama_functions import OllamaFunctions, _is_pydantic_class
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
import param
from langchain_core.output_parsers import JsonOutputParser
import os 
from langgraph.prebuilt import ToolNode
from langchain.tools.render import render_text_description
from langchain_community.llms import Ollama

load_dotenv()

class OllamaAgent(param.Parameterized):
    
    def __init__(self, tools, **params):
        super(OllamaAgent, self).__init__(**params)
        self.tools = render_text_description(tools)
        self.model = Ollama(model='mistral:instruct')
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key='input', output_key="output")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful but sassy assistant for Unipol service group. Your goal is to help the customer with their requests. \
                        To answer customer inquiries always use the tool available to you and NEVER make anything up. \
                        Answer only questions related to services offered by Unipol and never answer questions out of context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = create_tool_calling_agent(self.model, tool_chain, self.prompt)
        self.qa = AgentExecutor(agent=self.chain, tools=self.tools, verbose=False, memory=self.memory, return_intermediate_steps=False) 

    def message(self, query):
        if not query:
            return
        result = self.qa.invoke({"input": query})
        self.answer = result['output']
        return self.answer

    def clr_history(self, count=0):
        self.chat_history = []
        return
    
from operator import itemgetter

def tool_chain(self, model_output):
        tool_map = {tool.name: tool for tool in self.tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool

# def convert_to_ollama_tool(tool: Any) -> Dict:
#     """Convert a tool to an Ollama tool."""

#     if _is_pydantic_class(tool.__class__):
#         schema = tool.__dict__["args_schema"].schema()
#         definition = {"name": tool.name, "properties": schema["properties"]}
#         if "required" in schema:
#             definition["required"] = schema["required"]

#         return definition
#     raise ValueError(
#         f"Cannot convert tool to an Ollama tool. Needs to be a Pydantic model."
#     )
