from typing import Sequence, List, Union, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from json import JSONDecodeError
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain.agents.agent import MultiActionAgentOutputParser
import json


def create_tool_calling_agent(
    llm: BaseLanguageModel, tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Runnable:

    missing_vars = {"agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    if not hasattr(llm, "bind_tools"):
        raise ValueError(
            "This function requires a .bind_tools method be implemented on the LLM.",
        )
    llm_with_tools = llm.bind_tools(tools)

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_tool_messages(x["intermediate_steps"])
        )
        | prompt
        | llm_with_tools
        | ToolsAgentOutputParser()
    )
    return agent

class ToolAgentAction(AgentActionMessageLog):
    tool_call_id: str
    """Tool call that this message is responding to."""

def parse_ai_message_to_tool_action(
    message: BaseMessage,
) -> Union[List[AgentAction], AgentFinish]:
    """Parse an AI message potentially containing tool_calls."""
    if not isinstance(message, AIMessage):
        raise TypeError(f"Expected an AI message got {type(message)}")

    actions: List = []
    if message.tool_calls:
        tool_calls = message.tool_calls
    else:
        if not message.additional_kwargs.get("tool_calls"):
            return AgentFinish(
                return_values={"output": message.content}, log=str(message.content)
            )
        # Best-effort parsing
        tool_calls = []
        for tool_call in message.additional_kwargs["tool_calls"]:
            function = tool_call["function"]
            function_name = function["name"]
            try:
                args = json.loads(function["arguments"] or "{}")
                tool_calls.append(
                    ToolCall(name=function_name, args=args, id=tool_call["id"])
                )
            except JSONDecodeError:
                raise OutputParserException(
                    f"Could not parse tool input: {function} because "
                    f"the `arguments` is not valid JSON."
                )
    for tool_call in tool_calls:
        function_name = tool_call["name"]
        _tool_input = tool_call["args"]
        if "__arg1" in _tool_input:
            tool_input = _tool_input["__arg1"]
        else:
            tool_input = _tool_input

        content_msg = f"responded: {message.content}\n" if message.content else "\n"
        log = f"\nInvoking: `{function_name}` with `{tool_input}`\n{content_msg}\n"
        actions.append(
            ToolAgentAction(
                tool=function_name,
                tool_input=tool_input,
                log=log,
                message_log=[message],
                tool_call_id=tool_call["id"],
            )
        )
    return actions

class ToolsAgentOutputParser(MultiActionAgentOutputParser):
    @property
    def _type(self) -> str:
        return "tools-agent-output-parser"

    def parse_result(
            self, result: List[Generation], *, partial: bool = False
        ) -> Union[List[AgentAction], AgentFinish]:
            if not isinstance(result[0], ChatGeneration):
                raise ValueError("This output parser only works on ChatGeneration output")
            message = result[0].message
            return parse_ai_message_to_tool_action(message)

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        raise ValueError("Can only parse messages")


def _create_tool_message(
    agent_action: ToolAgentAction, observation: str
) -> ToolMessage:
    if not isinstance(observation, str):
        try:
            content = json.dumps(observation, ensure_ascii=False)
        except Exception:
            content = str(observation)
    else:
        content = observation
    return ToolMessage(
        tool_call_id=agent_action.tool_call_id,
        content=content,
        additional_kwargs={"name": agent_action.tool},
    )

def format_to_tool_messages(
    intermediate_steps: Sequence[Tuple[AgentAction, str]],
) -> List[BaseMessage]:
    messages = []
    for agent_action, observation in intermediate_steps:
        if isinstance(agent_action, ToolAgentAction):
            new_messages = list(agent_action.message_log) + [
                _create_tool_message(agent_action, observation)
            ]
            messages.extend([new for new in new_messages if new not in messages])
        else:
            messages.append(AIMessage(content=agent_action.log))
    return messages

