# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from simple_chatbot import SimpleChatbot

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # <-- allow requests from Live Server

bot = SimpleChatbot(kb_folder="kb")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    if not data or "message" not in data:
        return jsonify({"error": "missing 'message' in JSON body"}), 400
    user_message = data["message"]
    reply = bot.get_response(user_message)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

