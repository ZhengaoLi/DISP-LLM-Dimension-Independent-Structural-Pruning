DEBUG = False

if DEBUG:
    from torchviz import make_dot

import os
import time
import datetime
from functools import partial
from typing import Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from datasets import IterableDataset

import warnings
warnings.filterwarnings("ignore")

# Custom modules and tools
from utils import DistributedEnv
from data import dataloader_creator, load_hf_dataset_wikitext
from models import LlamaTokenizer, PruneLlamaForCausalLM, PruneLlamaDecoderLayer
from pruning import hypernetwork, collect_info_reg_llama, help_functions_hn


def round_to_block_size(current_rank, block_size=32):
    """Round `current_rank` down to the nearest multiple of `block_size`."""
    round_rank = max(block_size, (current_rank // block_size) * block_size)
    return round_rank


def main(
    exp_name: str = 'displlm',
    out_dir: str = None,
    hf_model: str = '/blue/sgao1/zhengao/Llama-2-7b-hf',
    learning_rate: float = None,
    total_n_step: int = 100000,
    start_iter: int = 0,
    batch_size: int = 1,
    use_fsdp: bool = True,
    num_workers: int = 2,
    rand_seed: int = None,
    non_hf_tokenizer_path: str = None,
    compile_flag: bool = True,
    p: float = 0.48,
    lam: float = 16.0,
    hn_block_size: int = 2048,
    hn_lr: float = 1e-3,
    min_hn_lr: float = 1e-3,
    use_sch: bool = False,
):

    # Initialize the distributed environment
    env = DistributedEnv()
    print(env)

    dist.init_process_group(
        backend="nccl",
        rank=env.global_rank,
        world_size=env.world_size,
        timeout=datetime.timedelta(seconds=3600 * 5),
    )

    # Use bf16 if supported, otherwise fallback to fp16
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Prepare output directory, random seed, and learning rate
    if out_dir is None:
        user_name = 'zhengao'
        dateTimeObj = datetime.datetime.now()
        out_dir = os.path.join('/output/', user_name, exp_name)

    if rand_seed is None:
        rand_seed = start_iter

    # Automatically calculate learning rate if not provided
    if learning_rate is None:
        llama_learning_rate_per_sample = 0.0003 / (4 * 1024 * 1024)
        learning_rate = min(
            llama_learning_rate_per_sample * batch_size * 4096 * env.world_size,
            0.0003
        )

    if env.global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    # Set the current GPU
    device_id = env.local_rank
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()

    # Load the tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer = hf_tokenizer
    if non_hf_tokenizer_path:
        env.print_master('Using non_hf_tokenizer ...')
        tokenizer = LlamaTokenizer(non_hf_tokenizer_path, output_type='list')
    ignored_token = tokenizer.bos_token_id

    # Load the prunable LLaMA model and collect pruning information
    model = PruneLlamaForCausalLM.from_pretrained(hf_model)
    model.config.use_cache = False
    env.print_master(model.config)
    print(model)
    
    # Load dataset
    tic = time.time()
    result_dataset = load_hf_dataset_wikitext('train', env.world_size * num_workers)

    train_dataloader_hn = dataloader_creator(
        dataset=result_dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        block_size=hn_block_size,
        num_workers=num_workers,
        cycling=False,
        rank=env.global_rank,
        world_size=env.world_size,
        ignored_token=ignored_token,
    )
    toc = time.time() - tic
    env.print(f"Initializing training dataset - done. Time: {toc:.2f}s")

    # Use collect_info_reg_llama to compute pruning regularization
    param_reg = collect_info_reg_llama(model, p=p, lam=lam)

    # Create hypernetwork for pruning
    hn = hypernetwork(t_structures=param_reg.structures)
    hn_helper = help_functions_hn(param_reg.structures)

    # Move the model and hypernetwork to the appropriate device
    hn.to(device_id)
    hn = DDP(hn)  # Wrap hypernetwork with DDP

    model.to(device_id)

    # Wrap with FSDP if enabled
    if use_fsdp:
        my_auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={PruneLlamaDecoderLayer}
        )

        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=data_type,
                reduce_dtype=data_type,
                buffer_dtype=data_type
            ),
        )
    else:
        model = DDP(model)

    # Enable torch.compile
    if compile_flag:
        model = torch.compile(model)

    # Train hypernetwork
    train_hn(
        env,
        model,
        hn=hn,
        train_hn_data=train_dataloader_hn,
        hn_helper=hn_helper,
        param_reg=param_reg,
        ignored_token=ignored_token,
        max_iter=total_n_step,
        out_dir=out_dir,
        p=p,
        hn_block_size=hn_block_size,
        hn_lr=hn_lr,
        min_hn_lr=min_hn_lr,
        use_sch=use_sch,
        use_fsdp=use_fsdp,
    )
    
def train_hn(
    env: DistributedEnv,
    model: torch.nn.Module,
    hn: torch.nn.Module,
    train_hn_data: IterableDataset,
    hn_helper,
    param_reg,
    start_iter=0,
    ignored_token=-1,
    log_interval=50,
    max_iter=100000,
    out_dir=None,
    p=None,
    hn_block_size=2048,
    hn_lr=1e-3,
    min_hn_lr=1e-3,
    use_sch=False,
    use_fsdp=False,
):
    data_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_id = env.local_rank
    iter_num = start_iter

    # Select the appropriate GradScaler (ShardedGradScaler for FSDP)
    if use_fsdp:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        scaler = ShardedGradScaler()
    else:
        scaler = GradScaler()

    # Enable learning rate scheduling if configured
    optimizer = torch.optim.AdamW(hn.parameters(), lr=hn_lr, weight_decay=0.05)
    if use_sch:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_iter,
            eta_min=min_hn_lr,
            last_epoch=iter_num - 1
        )

    tic = time.time()
    torch.cuda.empty_cache()

    # Freeze parameters of the main model
    for param in model.parameters():
        param.requires_grad = False
    # Enable training only for the hypernetwork
    for param in hn.parameters():
        param.requires_grad = True
    hn.train()

    while True:
        for batch in train_hn_data:
            if iter_num >= max_iter:
                break

            with torch.no_grad():
                input_ids = batch['input_ids'].to(device_id)
                targets = batch['labels'].to(device_id)
                input_ids = input_ids[:, :hn_block_size]
                targets = targets[:, :hn_block_size]

            with autocast(device_type='cuda', dtype=data_type):
                # Generate pruning vectors using the hypernetwork
                vectors = hn()
                hn_helper.set_gate_vectors(model, vectors) 

                # Forward pass
                output = model(input_ids)
                logits = output.logits if hasattr(output, 'logits') else output
                ce_loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=ignored_token
                )

                # Calculate pruning regularization
                if hasattr(hn, "module"):
                    hard_out = hn.module.hard_output()
                else:
                    hard_out = hn.hard_output()
                    
                reg_loss = param_reg(hard_out)

                total_loss = ce_loss + reg_loss

            # Check for NaN loss values
            if torch.isnan(total_loss):
                env.print_master("!!! nan loss detected !!!")
                total_loss.fill_(0)

            print("ce loss: %.4f" % ce_loss, "reg_loss: %.4f" % reg_loss)
            # Backward pass and optimizer step
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if use_sch:
                scheduler.step()

            if iter_num % log_interval == 0:
                elapsed = time.time() - tic
                env.print_master(
                    f"Iter {iter_num}/{max_iter}, "
                    f"loss={total_loss.item():.4f}, "
                    f"reg={reg_loss.item():.4f}, "
                    f"time={elapsed*1000:.2f}ms"
                )
                tic = time.time()

            iter_num += 1
            if iter_num >= max_iter:
                break
        if iter_num >= max_iter:
            break

    # Save the hypernetwork
    if env.world_size == 1:
        if env.global_rank == 0:
            torch.save(hn.state_dict(), os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt"))
    else:
        # For multi-GPU, save only on rank=0
        if hasattr(hn, "module"):
            state_dict_hn = hn.module.state_dict()
        else:
            state_dict_hn = hn.state_dict()

        if env.global_rank == 0:
            torch.save(state_dict_hn, os.path.join(out_dir, f"hn-ckpt-final-{p:.2f}.pt"))


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    from jsonargparse import CLI
    CLI(main)
