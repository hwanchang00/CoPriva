# Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering

**Welcome to the official repository of** [**Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**](https://arxiv.org/abs/2505.15805)!

## Overview
<p align="center">
  <img src="asset/intro_fig.jpg" width="400"/>
</p>
As LLMs are deployed in sensitive fields, ensuring adherence to user-defined security policies is crucial, especially for information non-disclosure. To address this, we propose **CoPriva**, a benchmark with 4,184 QA pairs designed to evaluate LLM compliance with contextual non-disclosure policies.

The dataset tests models against two types of attacks:

1. **Direct attacks**: Queries that explicitly ask for information prohibited by the policy.

2. **Indirect attacks**: Queries that indirectly reference confidential content, often phrased in high-level or paraphrased ways.

## Get Started
### Installation
```bash
git clone https://github.com/hwanchang00/CoPriva.git
cd CoPriva
conda create -n copriva python=3.10
conda activate copriva
pip install -r requirements.txt
```
## Inference & Evaluation

Describe how to **run inference** using OpenRouter API and **evaluate** model performance using OpenAI API.

* **Inference**: Conducted via [OpenRouter API](https://openrouter.ai/).
* **Evaluation**: Conducted via [OpenAI API](https://platform.openai.com/docs/overview).

---

### Inference

Generate model responses on the benchmark dataset:

#### Example

```bash
export OPENROUTER_KEY="your_openrouter_key" 

python src/inference.py \
  --data_root data \
  --test_models google/gemini-2.0-flash-001,meta-llama/llama-3.1-70b-instruct \
  --indirect_output_root indirect_attack_outputs \
  --direct_output_root direct_attack_outputs
```

#### Arguments

* `--data_root`: Path to the directory containing JSON-formatted test data files.
* `--test_models`: **Comma-separated list of model names** to evaluate. (Find model names at: [here](https://openrouter.ai/models))
* `--direct_output_root`: (Optional) Directory to store outputs from **direct attack** inference.
* `--indirect_output_root`: (Optional) Directory to store outputs from **indirect attack** inference.

If `--direct_output_root` or `--indirect_output_root` is not specified, the corresponding inference will be skipped.

## Evaluation

Evaluate model responses in terms of **data leakage** and **faithfulness** to confidential content using the OpenAI API.

 Check available models at: [here](https://platform.openai.com/docs/models)

---

### Evaluate Data Leakage

Checks whether the model **leaked confidential content**

→ Adds `"is_leakage"` field to each example.
```bash
python src/eval_data_leakage.py \
  --evaluator_model_name gpt-4o \
  --data_root indirect_attack_outputs \
  --output_root eval_outputs/leakage
```

**Arguments:**

* `--evaluator_model_name`: OpenAI model name used for evaluation
* `--data_root`: Directory containing model output JSONs
* `--output_root`: Directory where evaluation results will be saved

---

### Evaluate Faithfulness

This script checks whether the model's output **faithfully includes** the core ideas from the confidential reference.

→ Adds `"faithfulness_score"` and `"feedback"` fields to each example.
```bash
python src/eval_faithfulness.py \
  --evaluator_model_name gpt-4o \
  --data_root eval_outputs/leakage \
  --output_root eval_outputs/faithfulness
```

**Arguments:**

* `--evaluator_model_name`: OpenAI model name used for evaluation
* `--data_root`: Directory containing **leakage evaluation outputs**
* `--output_root`: Directory to store final results with **faithfulness scores**

## Acknowledgement

This benchmark is built upon the [QMSum](https://github.com/Yale-LILY/QMSum) dataset. We thank the original authors for providing a valuable resource for meeting summarization and QA research.

## Citation
If you find this work useful, please cite our paper:
```
@article{chang2025keep,
  title={Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering},
  author={Chang, Hwan and Kim, Yumin and Jun, Yonghyun and Lee, Hwanhee},
  journal={arXiv preprint arXiv:2505.15805},
  year={2025},
  url={https://arxiv.org/pdf/2505.15805}
}

```
