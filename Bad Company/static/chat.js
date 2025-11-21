// static/chat.js
const form = document.getElementById("input-form");
const input = document.getElementById("message-input");
const messages = document.getElementById("messages");

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = "message " + role;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  appendMessage("user", text);
  input.value = "";
  appendMessage("bot", "…thinking…");

  try {
    const resp = await fetch("http://127.0.0.1:5000/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    });
    const data = await resp.json();
    // remove last 'thinking' message
    const thinking = messages.querySelector(".message.bot:last-child");
    if (thinking && thinking.textContent === "…thinking…") {
      thinking.remove();
    }
    if (data.reply) {
      appendMessage("bot", data.reply);
    } else if (data.error) {
      appendMessage("bot", "Error: " + data.error);
    } else {
      appendMessage("bot", "No reply received.");
    }
  } catch (err) {
    appendMessage("bot", "Network error: " + err.message);
  }
});
