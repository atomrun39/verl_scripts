set -xeuo pipefail
#source /usr/local/Ascend/ascend-toolkit/set_env.sh
#source /usr/local/Ascend/nnal/atb/set_env.sh
export HCCL_EXEC_TIMEOUT=18000
export HCCL_CONNECT_TIMEOUT=1800
export GLOO_SOCKET_IFNAME=eno0
export TP_SOCKET_IFNAME=eno0
export HCCL_SOCKET_IFNAME=eno0
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:False,max_split_size_mb:128,garbage_collection_threshold:0.6"
# 使用v1引擎
export VLLM_USE_V1=1
# 指定vllm 版本
# export VLLM_VERSION=0.9.1

# 开启二级流水
export TASK_QUEUE_ENABLE=2
# 开启细绑核
export CPU_AFFINITY_CONF=1
# 使用jemalloc优化内存访问（依赖安装jemalloc）
export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"

export TIMESTAMP=$(date +"%m-%d_%H-%M")

# 机器单机8卡
trainer_n_gpus_per_node=8
trainer_nnodes=2
trainer_project_name='qwen3-grpo'
trainer_experiment_name='qwen3_4b_grpo_2node'-${TIMESTAMP}

RAY_DATA_HOME=${RAY_DATA_HOME:-"/data/cyd/verl"}
MODEL_PATH=${MODEL_PATH:-"/data/cyd/weights/Qwen3-4B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${trainer_project_name}/${trainer_experiment_name}"}
TRAIN_FILE=${TRAIN_FILE:-"/data/cyd/dapo-math-17k/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/cyd/dapo-math-17k/test.parquet"}

mkdir -p ./swanlab_logs/${trainer_project_name}
export SWANLAB_LOG_DIR=./swanlab_logs/${trainer_project_name}/${trainer_experiment_name}
export SWANLAB_MODE=cloud
mkdir -p "${RAY_DATA_HOME}/logs/${trainer_project_name}"
LOG_PATH="${RAY_DATA_HOME}/logs/${trainer_project_name}/${trainer_experiment_name}.log"

use_dynamic_bsz=True
max_prompt_length=2048
max_response_length=8192
max_tokens=$((max_prompt_length  + max_response_length))
log_prob_max_tokens=$((max_tokens * 2))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${TEST_FILE} \
    data.train_batch_size=32 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${log_prob_max_tokens} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${log_prob_max_tokens} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.use_torch_compile=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=${trainer_project_name} \
    trainer.experiment_name=${trainer_experiment_name} \
    trainer.logger=['console','swanlab'] \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.n_gpus_per_node=$trainer_n_gpus_per_node \
    trainer.nnodes=$trainer_nnodes \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False 2>&1 | tee ${LOG_PATH}