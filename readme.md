
# Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations

This repository contains the source code for [Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations](https://arxiv.org/abs/xxx) by **Yiyou Sun, Yu Gai, Lijie Chen, Abhilasha Ravichander, Yejin Choi, and Dawn Song**.

## 📌 Overview

This project investigates hallucinations in large language models (LLMs) by analyzing **subsequence associations** in model outputs. It provides tools for **benchmarking, evaluation, and visualization** of hallucinated content using the [HALoGEN](https://halogen-hallucinations.github.io/) dataset.

## ⚙️ Requirements

To set up the required environment, install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Running the Demo

You can test the hallucination analysis using the demo script:

`python demo.py`

The results will be stored in:

`halu_results/{model_name}/customize`

### 📊 Benchmark Results Reproduction

#### 📂 Data Source: HALoGEN

We use a preprocessed subset of HALoGEN prompts and corresponding hallucination subsequences. The preprocessed data is available in:

`./halu_results`

(The preprocess script is `python 1_data_prepare.py`)

#### 📈 Evaluation

To evaluate a model on the benchmark dataset:
	1.	Add your OpenAI API key to ./key.py.
	2.	Run the following command:

```
python 2_run_session_benchmark.py \
    --halo_type {dataset} \
    --model_name {model_name} \
    --num_perturbations 1024 \
    --min_level 2 \
    --search_beam 20 \
    --completion_modes openai-mask openai-token bert random \
    --test_sentence_num 25 \
    --eval_level_range quad
```

## 🔖 Citation

If you use our codebase, please cite our work:
```
@article{sun2025sat,
  title={Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations},
  author={Sun, Yiyou and Gai, Yu and Chen, Lijie and Ravichander, Abhilasha and Choi, Yejin and Song, Dawn},
  journal={arxiv},
  year={2025}
}
```