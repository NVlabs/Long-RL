# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import json
import os
import signal

import torch
from tqdm import tqdm

from llava.mm_utils import (
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.media import Video

# This function will be called when the timeout is reached
def handler(signum, frame):
    raise TimeoutError()


# Set the signal handler
signal.signal(signal.SIGALRM, handler)

def get_model_output(model, video_path, qs):
    prompt = []
    prompt.append(Video(video_path))
    prompt.append(qs)

    # Generate response
    response = model.generate_content(prompt)
    return response


def eval_model(args):
    # Model
    disable_torch_init()

    # List video files
    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    video_dir = args.video_dir
    video_files = os.listdir(video_dir)
    video_files = [f for f in video_files if os.path.splitext(f)[1] in video_formats]
    gt_questions = []
    for i, video_name in enumerate(video_files):
        gt_questions.append(
            {
                "video_name": video_name,
                "question": "Elaborate on the visual and narrative elements of the video in detail.",
            }
        )

    # Create the output directory if it doesn't exist
    args.output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    video_dir = os.path.expanduser(video_dir)
    print(f"Video directory: {video_dir}")

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name)

    # Iterate over each sample in the ground truth file
    index = 0
    for i, sample in tqdm(enumerate(gt_questions)):
        video_name = sample["video_name"]
        question = sample["question"]
        index += 1
        video_key = video_name.split(".")[0]
        caption_file = os.path.join(args.output_dir, "%s.json" % video_key)
        if os.path.exists(caption_file):
            print("Finished %s." % video_name)
            continue
        # Load the video file
        temp_path = os.path.join(video_dir, f"{video_name}")
        if not os.path.exists(temp_path):
            print(f"Video file not exist: {temp_path}")
            continue
        print(f"Processing video: {temp_path}")
        if os.path.exists(temp_path):
            video_path = temp_path
            try:
                output = get_model_output(model, video_path, question)
            except Exception:
                continue
            output_dict = {video_key: output}
            json.dump(output_dict, open(caption_file, "w"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/NVILA-8B-Video")
    parser.add_argument("--video_dir", type=str, default="", help="Directory that contains short video clips.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    args = parser.parse_args()

    eval_model(args)
