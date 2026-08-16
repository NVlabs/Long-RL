<p align="center" width="100%">
<img src="assets/long-rl-logo.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

# Long-RL: Scaling RL to Long Sequences

[![Paper](https://img.shields.io/badge/ArXiv-Paper-brown)](https://arxiv.org/abs/2507.07966)
[![Code](https://img.shields.io/badge/GitHub-Long%20RL-blue)](https://github.com/NVlabs/Long-RL)
[![Model](https://img.shields.io/badge/HuggingFace-Model-yellow)](https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B)
[![Video](https://img.shields.io/badge/YouTube-Video-red)](https://www.youtube.com/watch?v=ykbblK2jiEg)
[![Demo](https://img.shields.io/badge/Gradio-Demo-bron)](https://long-rl.hanlab.ai)

<div align="center">

[![Watch the video](assets/demo_video_first_frame.png)](https://www.youtube.com/watch?v=ykbblK2jiEg)

</div>

**Scaling RL to Long Videos [[Paper](https://arxiv.org/abs/2507.07966)]** <br />
[Yukang Chen](https://yukangchen.com/) *, [Wei Huang](https://aaron-weihuang.com/) *, [Baifeng Shi](https://bfshi.github.io/), [Qinghao Hu](https://tonyhao.xyz/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/), [Zhijian Liu](https://zhijianliu.com), [Pavlo Molchanov](https://www.pmolchanov.com/), [Jan Kautz](https://jankautz.com), [Xiaojuan Qi](https://xjqi.github.io/), [Sifei Liu](https://sifeiliu.net/),[Hongxu Yin](https://hongxu-yin.github.io/), [Yao Lu](https://scholar.google.com/citations?user=OI7zFmwAAAAJ&hl=en), [Song Han](http://songhan.mit.edu/) <br />

We introduce a full-stack framework that scales up reasoning in vision-language models (VLMs) to long videos, leveraging reinforcement learning. We address the unique challenges of long video reasoning by integrating three critical components: (1) a large-scale dataset, LongVideo-Reason, comprising 104K long video QA pairs with high-quality reasoning annotations across diverse domains such as sports, games, and vlogs; (2) a two-stage training pipeline that extends VLMs with chain-of-thought supervised fine-tuning (CoT-SFT) and reinforcement learning (RL); and (3) a training infrastructure for long video RL, named Multi-modal Reinforcement Sequence Parallelism (MR-SP), which incorporates sequence parallelism and a vLLM-based engine tailored for long video, using cached video embeddings for efficient rollout and prefilling. In our experiments, LongVILA-R1-7B achieves strong performance on video benchmarks, reaching 65.1% and 71.1% accuracy on VideoMME without and with subtitles, respectively, and consistently outperforming LongVILA-7B across multiple benchmarks. Moreover, LongVILA-R1-7B supports processing up to 8,192 video frames per video, and configurable FPS settings. Notably, our MR-SP system achieves up to 2.1x speedup on long video RL training. In addition, we release our training system for public availability that supports RL training on various modalities (video, text, and audio), various models (VILA and Qwen series), and even image and video generation models. On a single A100 node (8 GPUs), it supports RL training on hour-long videos (e.g., 3,600 frames).

<p align="center" width="100%">
<img src="assets/Efficiency-comparison2.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

| Models             | VideoMME (w/o sub) | VideoMME (w sub) | ActivityNet-QA (test) | LongVideoBench (val) | PerceptionTest (val) | NExT-QA (mc) | VNBench (val) |
|:-------------------|:------------------:|:----------------:|:---------------------:|:--------------------:|:--------------------:|:--------:|:-------------:|
| **LongVILA-7B**    |      **60.1**      |     **65.1**     |       **59.5**        |       **57.1**       |       **58.1**       | **80.7** |   **63.0**    |
| **LongVILA-R1-7B** |      **65.1**      |     **71.1**     |       **64.8**        |       **58.0**       |       **68.9**       | **81.5** |   **75.5**    |

## TABLE OF CONTENTS
1. [News](#news)
2. [Highlights](#highlights)
3. [Introduction](#introduction)
4. [LongVILA-R1 Model Usage](#longvila-r1-model-usage)
5. [Supported Features](#supported-features)
6. [Installation](#installation)
7. [Training](#training)
8. [LongVideo-Reason](#longvideo-reason)
9. [Examples](#examples)
10. [How to contribute](#how-to-contribute)
11. [Core Contributors](#core-Contributors)
12. [Citation](#citation)
13. [Acknowledgement](#acknowledgement)

## News
- [x] [2025.9.18] **Long-RL** has been accepted by NeurIPS 2025.
- [x] [2025.7.30] **LongVILA-R1-7B** supports processing up to **8,192** video frames per video, with configurable FPS settings. Please refer to its [usage instructions](#longvila-r1-model-usage).
- [x] [2025.7.24] We release a gradio demo (https://long-rl.hanlab.ai) with our LongVILA-R1-7B model deployed.
- [x] [2025.7.24] We release the model weights of **LongVILA-R1-7B** on HuggingFace (https://huggingface.co/Efficient-Large-Model/LongVILA-R1-7B). LongVILA-R1-7B achieves **65.1% / 71.1%** on VideoMME. It supports reasoning on both **multiple-choice** and **open-ended** questions, and can also switch to non-thinking mode.
- [x] [2025.7.19] We release a detailed instruction and scripts for the data generation process of our LongVideo-Reason dataset in the [`longvideo-reason`](longvideo-reason/) directory.
- [x] [2025.7.18] We release new supported features, including *Open-ended reward*, *Cached video embeddings*, and *Chunked gathering* as introduced in [Supported Features](#supported-features).
- [x] [2025.7.10] We release [Paper](https://arxiv.org/abs/2507.07966) and this GitHub repo [Long-RL](https://github.com/NVlabs/Long-RL).

## Highlights
1. **Hour-level long video RL training on a single node**:  We supports RL training on hour-level videos (3,600 frames - 256k tokens) with sequence parallel, on a single A100 node (8 GPUs). `examples/new_supports/qwen2_5_vl_3b_video_1h.sh`
2. **Omni-model RL**: We supports RL training on omni models, that take text, video, and audio for inputs. `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
3. **Image/video generation RL**:  We supports RL training on image/video generation models, like [Stable Diffusion](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) and [Wan](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) series models. `examples/new_supports/sd3_image_grpo.sh` and `examples/new_supports/wan_video_grpo.sh`.

## Introduction
**Support models**:
- [x] VILA series models on image and video, with SP support
  - `examples/new_supports/nvila_2b_clevr_grpo.sh`
  - `examples/new_supports/nvila_2b_video_grpo.sh`
  - `examples/new_supports/longvila_7b_video_grpo.sh`
- [x] Qwen-VL series models on text, image, video, and audio, with SP support
  - `examples/new_supports/qwen2_5_3b_math_grpo.sh`
  - `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
  - `examples/new_supports/qwen2_5_omni_3b_grpo.sh`
- [x] Image and video diffusion model RL
  - `examples/new_supports/sd3_image_grpo.sh`
  - `examples/new_supports/wan_video_grpo.sh`
    
**Support algorithms**:
- [x] In addition to GRPO, DAPO & Reinforce supported, with SP support
  - `examples/new_supports/qwen2_5_vl_3b_video_dapo.sh`
  - `examples/new_supports/qwen2_5_vl_3b_video_grpo.sh`
  - `examples/new_supports/qwen2_5_vl_3b_video_reinforce.sh`

## LongVILA-R1 Model Usage

### General Inference
```python
from transformers import AutoModel

model_path = "Efficient-Large-Model/LongVILA-R1-7B"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")

# You can adjust the FPS value as needed. 
# To disable FPS control, set it to 0 and manually specify the number of processed video frames via `num_video_frames`.
# Example:
# model.config.fps = 8.0
# model.config.num_video_frames, model.config.fps = 512, 0


use_thinking = True # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "What is the main purpose of the video?"
video_path = "video.mp4"

if use_thinking:
  prompt = system_prompt_thinking.format(question=prompt)

response = model.generate_content([prompt, {"path": video_path}])
print("Response: ", response)
```

### with vLLM engine
Tested on `vllm==0.9.1`. We need to get the remote code first.
```bash
mkdir remote_code
cp path_to/Efficient-Large-Model/LongVILA-R1-7B/*.py remote_code
```
Then, you can use the following code for model generation.
```python
import os
from transformers import AutoModel
from vllm import LLM, SamplingParams
from remote_code.media import extract_media
from remote_code.mm_utils import process_images
from remote_code.tokenizer_utils import tokenize_conversation

model_path = "path_to/Efficient-Large-Model/LongVILA-R1-7B"

model_encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto", llm_only_need_embed=True)
# you can change gpu_memory_utilization according to GPU memory
llm = LLM(model=os.path.join(model_path, "llm"), enable_prompt_embeds=True, gpu_memory_utilization=0.5)

use_thinking = True # Switching between thinking and non-thinking modes
system_prompt_thinking = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n\n Question: {question}"

prompt = "What is the main purpose of the video?"
video_path = "video.mp4"

if use_thinking:
  prompt = system_prompt_thinking.format(question=prompt)

conversation = [{"from": "human", "value": [prompt, {"path": video_path}]}]
media = extract_media(conversation, model_encoder.config)
input_ids = tokenize_conversation(conversation, model_encoder.tokenizer, add_generation_prompt=True).unsqueeze(0).cuda()
media["video"] = [
    process_images(images, model_encoder.vision_tower.image_processor, model_encoder.config).half()
    for images in media["video"]
]

inputs_embeds, _, _ = model_encoder._embed(input_ids, media, {"video": {}}, None, None)

completions = llm.generate(prompts=[{"prompt_embeds": inputs_embeds.squeeze(0)}], sampling_params=SamplingParams(max_tokens=1024))
response = completions[0].outputs[0].text
print("Response: ", response)
```


## Supported Features
- [x] **Open-ended reward**: 
- We support training for open-ended QAs (non-multi-choices QAs). Please do the following steps if you need it.
  - Set `--worker.rollout.open_ended_reward=True` in the training script.
  - Export your openai API with `export OPENAI_API_KEY=xxx`.
- [x] **Cached video embeddings**:
- We support using cached video embeddings for video RL training. Because video encoding during training is slow for large batch & long video frames. Please do the following steps if you neet it.
  - Follow `verl/utils/cache_video_embeds_vila.py` to cache video embeddings in a local directory.
  - Set `--data.cache_dir` and `--worker.actor.cached_embeds_dir` in the training script.
- [x] **Chunked gathering**:
- We support chunked gathering for `all_gather_data_proto`. Because it might suffer from CPU OOM if you machine do not have enough CPU memory, and also large batches or long video frames are needed. Please do the following step if you neet it.
  - Set `--worker.rollout.num_chunk_seq` in the training script. It can be 8/16/32. Larger ones cost less memory, but more time.

## Installation

```bash
git clone https://github.com/NVlabs/Long-RL.git
cd Long-RL
pip install -e .
```
If you want to train Qwen-Omni models, please
```bash
bash vllm_replace.sh
```

## Training
### Single node
For single node (within 8 GPUs), you can refer to the training scripts in the `examples` directory. For example,
```bash
bash examples/new_supports/qwen2_5_vl_3b_video_grpo.sh $VIDEO_PATH
```

### Multi-nodes
For jobs that requires multi-nodes, you can refer to the ways mentioned in the EasyR1 repo, [here](https://github.com/hiyouga/EasyR1/tree/main?tab=readme-ov-file#how-to-run-70b-model-in-multi-node-environment).

We provide additional examples for `sbatch` scripts like, where `TRAIN_SCRIPT` is the script to train on single node, `NNODES` is the number of nodes required.
```bash
bash scripts/srun_multi_nodes.sh $TRAIN_SCRIPT $NNODES
```

For example, 
```bash
bash scripts/srun_multi_nodes.sh examples/new_supports/qwen2_5_vl_3b_video_grpo.sh 2
```

### Merge Checkpoint in Hugging Face Format
This follows the ways in the EasyR1 repo.
```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/exp_name/global_step_1/actor
```

## LongVideo-Reason
We provide detailed instructions on the data generation process and how to evaluate models on our `LongVideo-Reason` benchmark in the [`longvideo-reason`](longvideo-reason/) directory. 

## Examples
<div align="center">
<a href="https://drive.google.com/file/d/1QJ-ZsDrmYS8v1XU4eWfYu5oHuXeyGSdK/view?usp=share_link">Football Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-football.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1U0N563a2s24o_NDie1VfWauxFuSu31wC/view?usp=share_link">Texas Hold’em Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-TexasHold.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1rnF4I6-EBpqhzA0SnwyajpxbAhMezDCn/view?usp=share_link">Starcraft II Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-starcraft2.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

<div align="center">
<a href="https://drive.google.com/file/d/1lo1E_bXXnMmWnFRudaSUgxMNxetEDHP9/view?usp=share_link">Moving Cup Video</a>
</div>
<p align="center" width="100%">
<img src="assets/example-movingcup.png" alt="Stanford-Alpaca" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>


## How to contribute
- Make sure to have git installed.
- Create your own [fork](https://github.com/NVlabs/Long-RL/fork) of the project.
- Clone the repository on your local machine, using git clone and pasting the url of this project.
- Read both the `Installation` sections above.
- Commit and push your changes.
- Make a pull request when finished modifying the project.


## Core Contributors
[Yukang Chen](https://yukangchen.com/), [Wei Huang](https://aaron-weihuang.com/), [Shuai Yang](https://andysonys.github.io), [Qinghao Hu](https://tonyhao.xyz/), [Baifeng Shi](https://bfshi.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/home), [Ligeng Zhu](https://lzhu.me/).

We welcome all possible contributions and will acknowledge all contributors clearly.

## Citation
Please consider to cite our paper and this framework, if they are helpful in your research.

```bibtex
@misc{long-rl,
  title = {Long-RL: Scaling RL to Long Sequences},
  author = {Yukang Chen, Wei Huang, Shuai Yang, Qinghao Hu, Baifeng Shi, Hanrong Ye, Ligeng Zhu, Zhijian Liu, Pavlo Molchanov, Jan Kautz, Xiaojuan Qi, Sifei Liu,Hongxu Yin, Yao Lu, Song Han},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/NVlabs/Long-RL}},
}
```
```bibtex
@inproceedings{chen2025longvila-r1,
      title={Scaling RL to Long Videos},
      author={Yukang Chen and Wei Huang and Baifeng Shi and Qinghao Hu and Hanrong Ye and Ligeng Zhu and Zhijian Liu and Pavlo Molchanov and Jan Kautz and Xiaojuan Qi and Sifei Liu and Hongxu Yin and Yao Lu and Song Han},
      booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
      year={2025},
}
```
```bibtex
@inproceedings{chen2024longvila,
      title={LongVILA: Scaling Long-Context Visual Language Models for Long Videos},
      author={Yukang Chen and Fuzhao Xue and Dacheng Li and Qinghao Hu and Ligeng Zhu and Xiuyu Li and Yunhao Fang and Haotian Tang and Shang Yang and Zhijian Liu and Ethan He and Hongxu Yin and Pavlo Molchanov and Jan Kautz and Linxi Fan and Yuke Zhu and Yao Lu and Song Han},
      booktitle={The International Conference on Learning Representations (ICLR)},
      year={2025},
}
```

## Acknowledgement
- [EasyR1](https://github.com/hiyouga/EasyR1): the codebase we built upon. Thanks for their wonderful work.
- [verl](https://github.com/volcengine/verl): the RL training framework we built upon.
- [vllm](https://github.com/vllm-project/vllm): we built upon vllm for the rollout engine.
- [Flow-GRPO](https://github.com/yifan123/flow_grpo): we refer to the Flow-GRPO for the image/video generation RL part.
