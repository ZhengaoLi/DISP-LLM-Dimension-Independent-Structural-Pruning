CUDA_VISIBLE_DEVICES=1,2 \
torchrun --nproc_per_node=2 train_hypernetwork.py \
    --hf_model /blue/sgao1/zhengao/Llama-2-7b-hf \
    --use_fsdp True \
    --p 0.48 \
    --lam 16.0 \
    --batch_size 1 \
    --total_n_step 10000 \
    --hn_lr 1e-3 \
    --min_hn_lr 1e-3 \
    --use_sch False \
    --out_dir ./output \
    --exp_name "PruneLlama"
