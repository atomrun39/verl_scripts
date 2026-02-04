docker run -it -u root --ipc=host --net=host --name=verl-test \
--device=/dev/davinci{0,1,2,3,4,5,6,7} \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin/:/usr/local/sbin/ \
-v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
-v /var/log/npu/slog/:/var/log/npu/slog \
-v /var/log/npu/profiling/:/var/log/npu/profiling \
-v /var/log/npu/dump/:/var/log/npu/dump \
-v /var/log/npu/:/usr/slog \
-v /usr/lib/jvm/:/usr/lib/jvm \
-v /data/:/data/ \
verl-8.3.rc1-910b-ubuntu22.04-py3.11:latest \
/bin/bash