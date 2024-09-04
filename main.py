from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from flask import Flask, request

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def create_app():
    app = Flask(__name__)

    llm = ChatVertexAI(
        model="gemini-1.5-flash-001",
        temperature=0.5,
        max_tokens=5000,
        max_retries=6,
        stop=None,
    )

    tools = [add]
    tools_dict = {"add": add}
    llm_with_tools = llm.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in answering questions about Japanese anime and able to give useful recommendations. Do not use emoji. Do not do math yourself. Always use tools for that.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | llm_with_tools
    
    history = ChatMessageHistory()

    @app.route("/", methods=['POST'])
    def talkToGemini():
        user_message = request.json['message']
        history.add_user_message(user_message)
        print(history.messages[-1])

        ai_response = chain.invoke({"messages": history.messages})

        history.add_ai_message(ai_response)
        print(history.messages[-1])

        if len(ai_response.tool_calls) == 0:
            return ai_response.content

        for tool_call in ai_response.tool_calls:
            selected_tool = tools_dict[tool_call["name"].lower()]
            tool_output = selected_tool.invoke(tool_call["args"])
            history.add_message(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
            print(history.messages[-1])

        ai_response = chain.invoke({"messages": history.messages})
        history.add_ai_message(ai_response.content)
        print(history.messages[-1])

        return ai_response.content

    return app

if __name__ == "__main__":
    print("Initializing, please wait...")
    app = create_app()
    app.run(host='0.0.0.0', port=8081)
