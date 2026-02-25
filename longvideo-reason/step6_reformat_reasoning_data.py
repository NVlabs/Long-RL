import os
import json
import openai
import requests
from argparse import ArgumentParser
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
except Exception:
    raise ValueError("Please set the environment variable OPENAI_API_KEY if you need open-ended reward computation.")

def _remove_captions(_input):
    _output = _input.replace("video captions", "video").replace("the captions", "the video").replace("The captions", "The video").replace("the video's captions", "the video").replace("The video's captions", "The video").replace("captions", "video frames").replace("caption", "video")
    return _output

def generate_gpt(prompt, model="gpt-4o"):
    PROMPT_MESSAGES = [

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

def convert_to_training_format(input_json, output_dir):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, item in enumerate(data):
        video_name = item["video_name"]
        output_path = os.path.join(output_dir, f"{video_name}.json")
        if os.path.exists(output_path):
            print(f"Skipping {video_name}")
            continue
        question = _remove_captions(item["QUESTION"])
        options = _remove_captions("\n".join([f"{key}. {value}" for key, value in item["OPTIONS"].items()]))
        answer = item["ANSWER"]

        # Construct reasons string
        reason = "Start thinking.\n" + "\n".join(
            [
                f"{step}: " + "\n".join(
                    details["reasons"])
                for step, details in item["REASONS"].items()
            ]
        )
        reason = _remove_captions(reason)

        prompt = f"""
            You are an advanced AI language model designed to refine logical reasoning while maintaining accuracy. Your task is to optimize the provided reasoning so that it is more natural, logically coherent, and easy to read. Ensure that the refined reasoning:
            
            1. Maintains all key information without introducing errors, while keeping the explanation detailed and avoiding any loss of information.
            2. Uses step-by-step formatting, and smooth logic.
            3. Removes unnecessary words like "Step" and time references such as (0:00:20–0:00:30).
            4. Incorporates a thoughtful and logical thinking process, especially when reasoning involves interpreting or searching within a video. Use phrases like "checking the video," "analyzing the scene," or "searching for specific actions or details in the video" to reflect a step-by-step exploration of the content.
            
            Here is the given input:
            
            "question": "{question}\n{options}"
            
            "answer": "{answer}"
            
            "reason": "{reason}"
            
            Please return only the optimized reasoning without any additional text or formatting. Ensure the output reflects a clear understanding of the video content and includes logical steps like reviewing or analyzing video details as necessary. The output should be in plain text, directly usable in a program.
        """
        reformated_reason = generate_gpt(prompt)

        if reformated_reason is None:
            continue
        formatted_data = {
            "problem_id": i,
            "data_type": "video",
            "videos": f"{video_name}.mp4",
            "problem": question,
            "problem_type": "general",
            "reasoning": reformated_reason,
            "answer": "<answer>%s</answer>"%answer,
        }

        json.dump(formatted_data, open(output_path, "w"), ensure_ascii=False, indent=4)
        print(f"Converted data saved to {output_path}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_json", type=str, default="path/to/reasoning_data.json")
    parser.add_argument("--output_dir", type=str, default="./output_dir")
    args = parser.parse_args()

    input_json = args.input_json
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    convert_to_training_format(input_json, output_dir)
