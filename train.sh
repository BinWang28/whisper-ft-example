





export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=~/scratch/huggingface
#export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0


torchrun --master_port 8442 --nproc_per_node=8 train.py 2>&1 | tee train.log

#CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port 8442 --nproc_per_node=2 train.py --base_model=openai/whisper-large-v2 --use_8bit=False --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --gradient_accumulation_steps=4

