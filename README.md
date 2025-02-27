### (NeurIPS-2024) DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models

---

####  Project Overview 

This project implements DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models. The code in this repo supports LLaMA 7B/13B and LLaMA-2 7B/13B models. The paper is availabe at https://arxiv.org/html/2410.11988v2.

---

####  Requirements

The code is extensively tested with Pytorch 2.0.1 and transformers 4.44.2, and a complete environment is provided in the environment.yml file.


####  Code Structure 
```
.
├── LICENSE.txt                # Licensing information
├── README.md                  # Project description
├── data/                      # Data processing utilities
│   ├── __init__.py
│   └── data_utils.py          # Functions for dataset loading and preprocessing
├── models/                    # Model architecture and tokenizer
│   ├── __init__.py
│   ├── modeling_llama_pruning.py  # LLaMA model with pruning support
│   └── tokenizer.py           # Tokenizer utilities for LLaMA
├── pruning/                   # Pruning and hypernetwork logic
│   ├── __init__.py
│   ├── hypernetwork.py        # Hypernetwork implementation
│   └── pruning_helper.py      # Helper functions for pruning
├── run1.sh                    # Example script for running training
├── train_hypernetwork.py      # Main script for hypernetwork training
└── utils/                     # General utility scripts
    ├── __init__.py
    └── distributed_env.py     # Utilities for distributed training environment
```
---

####  What This Code Does 

1.  Pruning with Hypernetwork :
   - The  hypernetwork  (`hypernetwork.py`) generates pruning vectors for each layer in LLaMA.

2.  Training Pipeline :
   - The main script (`train_hypernetwork.py`) handles training the hypernetwork while freezing the main LLaMA model.
   - The pruning process is guided by a  regularization loss  to enforce a target pruning ratio.

3.  Distributed Training Support :
   - The framework supports  Distributed Data Parallel (DDP)  and  Fully Sharded Data Parallel (FSDP)  for scaling across multiple GPUs or nodes.
   - Distributed environment setup is handled in `distributed_env.py`.

4.  Dataset Preprocessing :
   - Utilities in `data_utils.py` preprocess datasets, tokenize text, and create dataloaders.
   - Compatible with HuggingFace datasets, such as Wikitext.

5.  LLaMA Model with Pruning :
   - The LLaMA model architecture is adapted for pruning in `modeling_llama_pruning.py`.
   - Tokenization logic is provided in `tokenizer.py`.

---

#### How to Run


1. Multi-GPU Training :
   Use the `run1.sh` script for launching distributed training with torchrun or mpirun.
2. Model Pruning:
```
python prune_model.py --hf_model meta-llama/Llama-2-7b-hf --hn_path path/to/your/hn/hn-ckpt-final-0.50.pt --out_dir path/to/your/out_dir
```
3. Evaluate Zero-Shot Performance with lm-evaluation-harness. Make sure you have installed lm-evaluation-harness following their [installation guide](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 12323 --num_processes 1 \
    -m lm_eval --model hf \
    --model_args pretrained=/path/to/your/out_dir/,dtype="bfloat16",trust_remote_code=true \
    --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande \
    --batch_size 16 
```
4. If you want to perform fine-tuning, simply treat the model files located in `path/to/your/out_dir` as as standard Hugging Face models. You can use the existing PEFT library or any other fine-tuning libraries for this purpose.

---
#### To-Do List
- Add supports for other datasets and models.

#### Citation
```
@inproceedings{gaodisp,
  title={DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models},
  author={Gao, Shangqian and Lin, Chi-Heng and Hua, Ting and Tang, Zheng and Shen, Yilin and Jin, Hongxia and Hsu, Yen-Chang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
