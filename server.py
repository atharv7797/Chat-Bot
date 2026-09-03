from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("message", "")

    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-v4-pro-0813",
        messages=[
            {"role": "user", "content": user_text}
        ],
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        seed=42,
        extra_body={
            "chat_template_kwargs": {
                "thinking": False
            }
        },
        stream=False
    )

    reply = completion.choices[0].message.content

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)