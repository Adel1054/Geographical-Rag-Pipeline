import os
import requests
import json
from app.config import LLM_MODEL
from dotenv import load_dotenv

load_dotenv()

def build_prompt(question, context_chunks):
    context = "\n\n".join([chunk["text"] for chunk in context_chunks])
    return f"""You are an expert geographical assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say "I don't have enough information."

<context>
{context}
</context>

<question>
{question}
</question>

Answer:"""

def query_llm(prompt):
    headers = {
        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.3,
        "top_p": 0.9
    }
    res = requests.post("https://api.together.xyz/inference", headers=headers, json=payload)

    try:
        response = res.json()
        print("\nRaw Response:", response)

        return response["output"]["choices"][0]["text"].strip()
    except Exception as e:
        raise ValueError(f"Failed to parse response: {res.text}")