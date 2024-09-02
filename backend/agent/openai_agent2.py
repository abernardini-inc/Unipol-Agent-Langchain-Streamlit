from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI 
from langchain.agents import AgentExecutor
from langchain import hub
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.tools.render import render_text_description
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
        print("Inizializzazione Agent OpenAi 2")
        self.model = ChatOpenAI(model = "gpt-4-1106-preview", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.system_prompt = read_system_prompt(f"{os.path.dirname(__file__)}/prompt/system_prompt.txt")
        
        self.system_prompt = self.system_prompt.replace("{tools}", render_text_description(tools))
        self.system_prompt = self.system_prompt.replace("{tool_names}", ", ".join([tool.name for tool in tools]))
        print(self.system_prompt)
        # self.system_prompt.format(
        #     tools=render_text_description(tools),
        #     tool_names=", ".join([t.name for t in tools]),
        # )
        self.llm_with_stop = self.model.bind(stop=["\nObservation"])
        self.chain = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.system_prompt
            | self.llm_with_stop
            | ReActSingleInputOutputParser()
        )

        self.qa = AgentExecutor(agent=self.chain, tools=tools, verbose=False, memory=self.memory) 

    def message(self, query):
        if not query:
            return
        result = self.qa.invoke({"input": query})
        self.answer = result["output"]
        return self.answer

    def clr_history(self, count=0):
        print("\nEliminazione memoria")
        self.chat_history = []
        self.memory.clear()
        return
