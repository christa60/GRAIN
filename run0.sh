export HF_HOME="/project/lw29/ntb23"
export HF_HUB_CACHE="/project/lw29/ntb23"


for SEED in 42 52 62 72 82 92
do  
    echo $SEED
    accelerate launch finetune_pubmed_qwen2.py \
        --model_name Qwen/Qwen2-7B-Instruct \
        --output_dir ./qwen2-pubmedqa/acc/$SEED \
        --num_train_epochs 10 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --use_wandb False \
        --seed $SEED
done