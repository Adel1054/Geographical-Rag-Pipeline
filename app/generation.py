import os
from together import Together
from dotenv import load_dotenv

load_dotenv("key.env")
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


def build_prompt(query: str, context: str) -> str:
    """Constructs the prompt for the LLM using XML boundaries."""
    return f"""You are a helpful assistant. Use the following context to answer the user's question.

<context>
{context}
</context>

<question>
{query}
</question>
"""


def query_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant providing accurate geographical information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with Together AI: {e}")
        return "Sorry, I am currently unable to connect to the AI service."