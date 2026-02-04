#/bin/bash
MODEL_NAME=Qwen/Qwen3-4B
local_dir=Qwen3-4B
LOCAL_DIR=/data/cyd/weights/${local_dir}
modelscope download --model ${MODEL_NAME} --local_dir ${LOCAL_DIR}