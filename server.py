from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_text = data.get("message", "")

    def generate():
        stream = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro-0813",
            messages=[
                {"role": "user", "content": user_text}
            ],
            temperature=1,
            top_p=0.95,
            max_tokens=2048,
            seed=42,
            extra_body={
                "chat_template_kwargs": {
                    "thinking": False
                }
            },
            stream=True
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return Response(
        stream_with_context(generate()),
        content_type="text/plain; charset=utf-8"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)