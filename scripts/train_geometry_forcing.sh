# === training start ===
date=$(date +%Y%m%d_%H%M%S)
output_dir="output"
exp_name="dfot-geometry-forcing-re10k-16f-${date}"
result_dir="$output_dir/train/$exp_name"
init_ckpt_path="checkpoints/DFoT_16f_state_dict.ckpt"
eval_result_dir="$output_dir/evaluations/$exp_name"
algorithm="dfot_geometry_forcing"

mkdir -p "$result_dir" "$eval_result_dir"
echo "output_dir: ${output_dir}"

# === auto resume config ===
checkpoints_dir="${result_dir}/checkpoints"
echo "checkpoints_dir: ${checkpoints_dir}"

# Find the latest checkpoint (.ckpt file), if any
latest_ckpt=$(ls -t "${checkpoints_dir}"/*.ckpt 2>/dev/null | head -n 1)

if [ -n "${latest_ckpt}" ] && [ -e "${latest_ckpt}" ]; then
  # Resolve absolute path
  latest_ckpt_abs=$(realpath "${latest_ckpt}")
  echo "Latest checkpoint found: ${latest_ckpt_abs}"
  # Update init_ckpt_path to latest checkpoint
  init_ckpt_path="${latest_ckpt_abs}"
  echo "init_ckpt_path is set to latest checkpoint: ${init_ckpt_path}"
else
  echo "No checkpoint found in ${checkpoints_dir}."
  echo "init_ckpt_path remains as: ${init_ckpt_path}"
fi

# === training command ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

python -m main +name=RE10k dataset=realestate10k \
        algorithm=$algorithm \
        experiment=video_generation @diffusion/continuous \
        algorithm.alignment.latents_info=2 \
        algorithm.alignment.alignment_coeff=0.1 \
        algorithm.alignment.joint_finetune_vggt=true \
        algorithm.alignment.ema_momentum=0.995 \
        algorithm.alignment.vggt_distill_coeff=1.0 \
        algorithm.alignment.alignment_context_length=16 \
        algorithm.alignment.enable_vggt_checkpoint=true \
        algorithm.alignment.vggt_checkpoint_mode=block \
        algorithm.alignment.vggt_checkpoint_use_reentrant=false \
        load=$init_ckpt_path \
        dataset.subdataset_size=10000 \
        experiment.training.lr=8e-6 \
        experiment.training.max_epochs=24 \
        experiment.training.batch_size=1 \
        experiment.training.optim.accumulate_grad_batches=20 \
        algorithm.backbone.use_checkpointing=[true,true,true,true] \
        ++experiment.training.checkpointing.save_top_k=1 \
        ++experiment.training.checkpointing.save_weights_only=true \
        experiment.validation.batch_size=2 \
        experiment.training.compile=False \
        experiment.training.data.num_workers=16 \
        hydra.run.dir=$result_dir 

# === evaluation start ===
## construct evaluation args 
checkpoints_dir=${result_dir}/checkpoints
# find the latest checkpoint 
latest_ckpt=$(ls -t ${checkpoints_dir}/*.ckpt | head -n 1)
echo "Latest checkpoint: ${latest_ckpt}"
# get the absolute path of the latest checkpoint 
latest_ckpt_abs=$(realpath ${latest_ckpt})

if [ -z "$latest_ckpt_abs" ]; then
  echo "No checkpoint found in ${checkpoints_dir}. Exiting."
  exit 1
fi

echo "Absolute path of the latest checkpoint: ${latest_ckpt_abs}"
# = is not allowed to pass to eval cmd 
lasted_ckpt_path=${checkpoints_dir}/latest.ckpt
ln -s ${latest_ckpt_abs} ${lasted_ckpt_path}

python -m main +name=single_image_to_long dataset=realestate10k \
        algorithm=$algorithm experiment=video_generation \
        @diffusion/continuous \
        load=$lasted_ckpt_path \
        'experiment.tasks=[validation]' experiment.validation.data.shuffle=False experiment.test.data.shuffle=False \
        dataset.context_length=1 dataset.frame_skip=1 dataset.n_frames=256 \
        algorithm.tasks.prediction.keyframe_density=0.0625 \
        algorithm.tasks.interpolation.max_batch_size=4 experiment.validation.batch_size=1 \
        algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla \
        +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0 \
        +algorithm.tasks.prediction.history_guidance.stabilization_level=0.02  \
        algorithm.tasks.interpolation.history_guidance.name=vanilla \
        +algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5 \
        'algorithm.logging.metrics=[fvd,fid,psnr,lpips,ssim]' \
        hydra.run.dir=$eval_result_dir 