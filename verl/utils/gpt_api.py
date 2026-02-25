import openai
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    print("Please set the environment variable OPENAI_API_KEY if you need open-ended reward computation.")

def generate_gpt(prompt, model="gpt-4o-mini-2024-07-18"):
    PROMPT_MESSAGES = [
        #{"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text","text": prompt,},
            ]
        }
    ]
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=PROMPT_MESSAGES,
            max_tokens=2000,
            temperature=0.7
        )
        output = response.choices[0].message.content
        return output
    except Exception as e:
        print("Error", e)
        return None

