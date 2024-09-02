from fastapi import FastAPI
from pydantic import BaseModel
from tools.customer_tools import get_customer_info, get_customer_bills, sign_contract
from tools.retriver_tools import get_all_info
from agent.openai_agent import OpenAiAgent
from agent.claude_agent import ClaudeAgent

app = FastAPI()

tools = [get_all_info, get_customer_info, get_customer_bills, sign_contract]

agent = ClaudeAgent(tools)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/send_query")
async def send_query(query: QueryRequest) -> str:
    response = agent.message(query.query)
    return response

@app.get("/new_chat")
async def new_chat():
    agent.clr_history()
    return {"message": "New chat session started"}

