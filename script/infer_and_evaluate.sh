#!/bin/bash

export OPENROUTER_KEY="openrouter_key" 
export OPENAI_API_KEY="openai_api_key"

# Inference on the test set using the specified models.
python src/inference.py \
  --data_root data \
  --test_models google/gemini-2.0-flash-001,meta-llama/llama-3.1-70b-instruct \
  --indirect_output_root indirect_attack_outputs \
  --direct_output_root direct_attack_outputs \

# Perform data_leakage evaluation on the inference output.
python src/eval_data_leakage.py \
  --evaluator_model_name gpt-4.1-2025-04-14 \
  --data_root models/output/root \
  --output_root eval/output/root 

# Re-run the faithfulness evaluation on the evaluation root in data_leakage.
  python src/eval_faithfulness.py \
  --evaluator_model_name gpt-4.1-2025-04-14 \
  --data_root eval/output/root \
  --output_root eval/output/root