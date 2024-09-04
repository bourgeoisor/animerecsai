from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from flask import Flask, request
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    chat = ChatVertexAI(
        model="gemini-1.5-flash-001",
        temperature=0.5,
        max_tokens=5000,
        max_retries=6,
        stop=None,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert in answering questions about Japanese anime and able to give useful recommendations. Do not use emoji.

                You have access to the following function, which you should use as much as possible:
                - get_anime_info
                  params: [anime_name]

                To call this function, simply respond with this format, and nothing else:
                    FUNCTION: get_anime_info
                    PARAMS:
                        anime_name: PLACEHOLDER

                Else, reply as normal.""",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | chat

    history = ChatMessageHistory()

    @app.route("/", methods=['POST'])
    def talkToGemini():
        user_message = request.json['message']
        history.add_user_message(user_message)
        print("> USER")
        print("> " + user_message)

        ai_response = chain.invoke({"messages": history.messages})

        if ai_response.content.startswith("{"):
            print("> AI")
            print(ai_response.content)
            return ""

        history.add_ai_message(ai_response.content)
        print("> AI")
        print("> " + ai_response.content)

        return ai_response.content

    return app

if __name__ == "__main__":
    print("Initializing, please wait...")
    app = create_app()
    app.run(host='0.0.0.0', port=8081)
