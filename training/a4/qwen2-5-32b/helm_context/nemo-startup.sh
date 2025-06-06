#!/bin/bash

function on_script_completion {
  # Note: This semaphore is used to terminate the TCPx side-car
  touch /semaphore/workload_terminated
}
trap on_script_completion EXIT
trap "" SIGPROF

echo "Pod on $(hostname --fqdn) is running"
echo "Pod is assigned job index of $JOB_COMPLETION_INDEX"
echo "Job ID is $JOB_IDENTIFIER"

echo "The following GPUs are visible via nvidia-smi:"
nvidia-smi --list-gpus

echo "The following GPUs are visible via nvidia-smi:"
nvidia-smi --list-gpus

# Note: Including this prevented some past errors. It might become deprecated in future.
mount /tmp -o remount,exec 
chmod -R a+rwx /tmp

touch $SSD_MOUNT_PATH/hello-from-$HOSTNAME.txt
echo "Local SSD contents (path $SSD_MOUNT_PATH):"; ls $SSD_MOUNT_PATH | sed 's/^/  /'

echo "Contents (mounted at /usr/local/nccl-plugin/):"
ln -s /usr/local/nccl-plugin /usr/local/gib

export LD_LIBRARY_PATH="/usr/local/gib/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"
echo "Warning: Set LD_LIBRARY_PATH=$LD_LIBRARY_PATH to override the NCCL library"

source /usr/local/gib/scripts/set_nccl_env.sh

ldconfig /usr/local/nvidia/lib64/
echo "Added /usr/local/nvidia/lib64/ to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'

if ! [ -z ${JIT_GCS_FUSE_BUCKET} ]; then
  echo "Got request to JIT mount GCS bucket $JIT_GCS_FUSE_BUCKET via 'gcsfuse' to $JIT_GCS_FUSE_MOUNT_PATH:"
  mkdir -p $JIT_GCS_FUSE_MOUNT_PATH
  gcsfuse --client-protocol http2 $JIT_GCS_FUSE_BUCKET $JIT_GCS_FUSE_MOUNT_PATH 
fi

# It's for the GPT-2 tokenization scheme. To be removed in future.
echo "Downloading GPT vocabulary files"
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json &&\
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt

echo "NeMo configuration file:"                                         
cat /etc/workload-configuration/nemo-configuration.yaml | sed 's/^/| /' 
echo ""                                                                                                                                                
readarray -d "" workload_arguments < <(env | grep -e "^WORKLOAD_" | sed 's/^WORKLOAD_/+/' | tr '\n' '\0') 
echo "Detected the following additional workload arguments:"            
for workload_argument in "${workload_arguments[@]}"; do                 
  echo "  $workload_argument"                                           
done 

sleep 10 # <- Hack to allow some time for service to boot

echo "Checking for presence of nsys:"                                   
which nsys  

echo "NeMo job artifacts will go to /gcs/nemo-experiments/$JOB_IDENTIFIER/"
mkdir -p /gcs/nemo-experiments/$JOB_IDENTIFIER/

export NODE_RANK=$JOB_COMPLETION_INDEX                                  
if [ "$NODE_RANK" -eq "0" ] && { ! [ -z ${EMBEDDED_TENSORBOARD_TARGET} ]; }; then
  echo "Launching an embedded Tensorboard against log directory $EMBEDDED_TENSORBOARD_TARGET"
  tensorboard --logdir $EMBEDDED_TENSORBOARD_TARGET &
fi

export NODE_RANK=$JOB_COMPLETION_INDEX 
export WORLD_SIZE=$WORLD_SIZE
export TOKENIZERS_PARALLELISM=false
echo "Launching Torch distributed as node rank $NODE_RANK out of $NNODES nodes"
for ((LOCAL_RANK=0; LOCAL_RANK <= $((GPUS_PER_NODE - 1)); LOCAL_RANK++)); do
  RANK=$((8*$NODE_RANK + $LOCAL_RANK))

  OMP_NUM_THREADS=12 RANK=$RANK LOCAL_RANK=$LOCAL_RANK \
    nsys profile -s none -t nvtx,cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o /gcs/nemo-experiments/$JOB_IDENTIFIER/rank-$RANK \
    --force-overwrite true --session-new "nsys-$JOB_IDENTIFIER-$RANK" \
    python $TORCH_DISTRIBUTED_TARGET \
    --config-path="/etc/workload-configuration" \
    --config-name="nemo-configuration.yaml" \
    +trainer.num_nodes="$NNODES" \
    +exp_manager.version="$JOB_IDENTIFIER" \
    ${workload_arguments[@]} &

  echo "Launched rank $RANK with PID $!"
  TORCH_PIDS[$LOCAL_RANK]=$!                                            
done  

if [ "$NODE_RANK" -eq "1" ]; then
    echo "Launching nvidia-smi in daemon mode with (20 sec delay)"
    nvidia-smi dmon -d 20 -s pum &
fi

if [ "$NODE_RANK" -eq "0" ] && { ! [ -z ${EMBEDDED_TENSORBOARD_TARGET} ]; }; then
  echo "Launching an embedded Tensorboard against log directory $EMBEDDED_TENSORBOARD_TARGET"
  tensorboard --logdir $EMBEDDED_TENSORBOARD_TARGET &
  wait # <-- This will indefinitely stall node rank 0
fi

# Wait for Torch processes (might be problematic if only one fails)
for PID in ${TORCH_PIDS[*]}; do
  echo "Waiting on Torch PID $PID"
  wait $PID
done
echo "Pod on $(hostname --fqdn) is exiting"
