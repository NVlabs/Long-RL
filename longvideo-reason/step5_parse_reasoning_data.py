import os
import json
import re
from argparse import ArgumentParser


def parse_reasoning(reasoning_text):
    parsed_data = {}

    # Extract QUESTION
    question_match = re.search(r"QUESTION:\s*(.*)", reasoning_text)
    parsed_data["QUESTION"] = question_match.group(1) if question_match else ""

    # Extract OPTIONS
    options_match = re.findall(r"([A-D])\.\s*(.*)", reasoning_text)
    parsed_data["OPTIONS"] = {opt: text for opt, text in options_match}

    # Extract ANSWER
    answer_match = re.search(r"ANSWER:\s*([A-D])", reasoning_text)
    parsed_data["ANSWER"] = answer_match.group(1) if answer_match else ""

    # Extract REASONS
    reasons = {}
    if "##### From [" in reasoning_text:
        reason_blocks = re.split(r"##### From \[.*?\]", reasoning_text)[1:]
        reason_blocks_2 = re.split(r"##### From ", reasoning_text)[1:]
    else:
        reason_blocks = re.split(r"##### From .*?\n", reasoning_text)[1:]
        reason_blocks_2 = re.split(r"##### From ", reasoning_text)[1:]

    for i, block in enumerate(reason_blocks):
        if block[0] == ":":
            block = block[1:]
        step_reasons = [line.strip('- ') for line in block.strip().split('\n') if line.startswith('- ')]
        try:
            timestamp = reason_blocks_2[i].split(block)[0].split("[")[1].split("]")[0]
        except Exception:
            timestamp = ""
        reasons[f"Step {i + 1}"] = {"timestamp": timestamp, "reasons": step_reasons}

    parsed_data["REASONS"] = reasons

    return parsed_data


def process_json_files(folder_path, output_file):
    all_parsed_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                try:
                    if "video_name" in data and "reasoning" in data:
                        parsed_info = parse_reasoning(data["reasoning"])
                        parsed_info["video_name"] = data["video_name"]
                        all_parsed_data.append(parsed_info)
                except Exception:
                    print("Failed to process %s"%file_path)
    # Save parsed data to a new JSON file
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(all_parsed_data, out_f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="path/to/the_generated_reasoning_data")
    parser.add_argument("--output_file", type=str, default="parsed_reasoning_data.json")
    args = parser.parse_args()

    process_json_files(args.input_dir, args.output_file)
    print(f"Parsed data saved to {args.output_file}")
