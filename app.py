import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from anthropic import Anthropic

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

def get_client():
    if not API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY chưa được cấu hình")
    return Anthropic(api_key=API_KEY)

@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "AI backend"})

@app.post("/analyze")
def analyze():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()

    if not prompt:
        return jsonify({"ok": False, "error": "Thiếu prompt"}), 400

    try:
        client = get_client()
        message = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        text = "".join(
            block.text for block in message.content
            if getattr(block, "type", "") == "text"
        )

        return jsonify({
            "ok": True,
            "result": text
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
