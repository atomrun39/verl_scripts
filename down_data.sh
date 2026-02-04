#/bin/bash
DATA_NAME=AI-ModelScope/DAPO-Math-17k
local_dir=DAPO-Math-17k
LOCAL_DIR=/data/cyd/dataset/${local_dir}
modelscope download --dataset ${DATA_NAME} --local_dir ${LOCAL_DIR}