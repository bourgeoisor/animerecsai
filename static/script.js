const sendButton = document.getElementById("sendButton");
const inputMessage = document.getElementById("inputMessage");
const messagesDiv = document.getElementById("messages");

async function main() {
    sendButton.addEventListener("click", handleButtonClick);
    inputMessage.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
        sendButton.click();
        }
    });
    inputMessage.focus();
}

async function handleButtonClick() {
    if (!inputMessage.value || !inputMessage.value.trim()) {
        return;
    }

    const message = inputMessage.value;
    message.value = "";
    console.log("USER: " + message)

    const humanMessageDiv = document.createElement("div");
    humanMessageDiv.classList.add("message", "human-message");
    humanMessageDiv.innerText = message;
    messagesDiv.appendChild(humanMessageDiv);

    sendButton.disabled = true;
    inputMessage.disabled = true;
    inputMessage.value = "";

    const AIMessageDiv = document.createElement("div");
    AIMessageDiv.classList.add("message", "ai-message", "ai-message-loading");
    AIMessageDiv.innerText = "";
    messagesDiv.appendChild(AIMessageDiv);

    const response = await fetch("/llm", {
        method: "POST",
        headers: {
        "Content-Type": "application/json",
        },
        body: JSON.stringify({
        message: message
        }),
    });
    const responseText = (await response.text()).trim();
    console.log("AI: " + responseText)

    AIMessageDiv.innerText = responseText;
    AIMessageDiv.classList.remove("ai-message-loading");

    sendButton.disabled = false;
    inputMessage.disabled = false;
    inputMessage.focus();
}

main();