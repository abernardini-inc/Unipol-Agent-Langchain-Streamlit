from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import render_text_description
from langchain.agents import AgentExecutor
from langchain.schema.runnable import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import param
import os 

load_dotenv()

def read_system_prompt(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
class OpenAiAgent(param.Parameterized):
    def __init__(self, tools, **params):
        super(OpenAiAgent, self).__init__(**params)
        print("INFO:    Inizializzazione Agent OpenAi")
        self.functions = [format_tool_to_openai_function(f) for f in tools]
        self.model = ChatOpenAI(model = "gpt-4-1106-preview", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"]).bind(functions=self.functions)
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history", input_key='input', output_key="output")
        self.system_prompt = read_system_prompt(f"{os.path.dirname(__file__)}/prompt/system_prompt2.txt")
        self.system_prompt = self.system_prompt.replace("{tools}", render_text_description(tools))
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        self.chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | self.prompt | self.model | OpenAIFunctionsAgentOutputParser()
        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory, return_intermediate_steps=True) 

    def message(self, query):
        if not query:
            return
        result = self.qa.invoke({"input": query})
        self.answer = result['output']
        return self.answer

    def clr_history(self, count=0):
        print("\nINFO:  Eliminazione memoria")
        self.chat_history = []
        self.memory.clear()
        return
