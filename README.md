### Dimension-Independent Structural Pruning for Large Language Models

---

####  Subject

Computation and Language; Machine Learning

---

####  Project Overview 

This project implements a  hypernetwork-based pruning approach  for the LLaMA language model, enabling efficient pretraining with reduced parameters while maintaining performance. The hypernetwork dynamically generates pruning vectors for different layers of the model, applying structured pruning to reduce the computational cost.

---

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

1.  Dynamic Pruning with Hypernetwork :
   - The  hypernetwork  (`hypernetwork.py`) generates pruning vectors dynamically for each layer in LLaMA.

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


1.  Multi-GPU Training :
   Use the `run1.sh` script for launching distributed training with torchrun or mpirun.

---

####  Key Features 

-  Dynamic Pruning : Reduces model parameters while preserving performance.
-  Distributed Training : Scales efficiently across GPUs and nodes.
-  Customizable Pipeline : Easily adjust pruning ratios, learning rates, and block sizes.
---
#### To-Do List
- Perform pruning and evaluation on the LLaMA-2 7B model using the trained hypernetwork weights.
#### Citation
```
@inproceedings{gaodisp,
  title={DISP-LLM: Dimension-Independent Structural Pruning for Large Language Models},
  author={Gao, Shangqian and Lin, Chi-Heng and Hua, Ting and Tang, Zheng and Shen, Yilin and Jin, Hongxia and Hsu, Yen-Chang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
