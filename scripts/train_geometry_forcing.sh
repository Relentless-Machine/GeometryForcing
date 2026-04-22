# === training start ===
date=$(date +%Y%m%d_%H%M%S)
output_dir="output"
base_exp_name="${BASE_EXP_NAME:-dfot-geometry-forcing-re10k-16f}"
use_fixed_exp_name="${USE_FIXED_EXP_NAME:-false}"
auto_resume_wandb_id="${AUTO_RESUME_WANDB_ID:-false}"
if [ "${use_fixed_exp_name}" = "true" ]; then
  exp_name="${base_exp_name}"
else
  exp_name="${base_exp_name}-${date}"
fi
result_dir="$output_dir/train/$exp_name"
init_ckpt_path="${INIT_CKPT_PATH:-checkpoints/geometry_forcing_state_dict.ckpt}"
eval_result_dir="$output_dir/evaluations/$exp_name"
algorithm="dfot_geometry_forcing"
wandb_resume_arg=""
resolved_wandb_resume_id="${WANDB_RESUME_ID:-}"
if [ -z "${resolved_wandb_resume_id}" ] \
  && [ "${use_fixed_exp_name}" = "true" ] \
  && [ "${auto_resume_wandb_id}" = "true" ]; then
  # Reuse the latest run id under this fixed experiment directory.
  # Expected run dir names: run-YYYYMMDD_HHMMSS-<id> or offline-run-YYYYMMDD_HHMMSS-<id>
  latest_wandb_run_dir=$(ls -dt "${result_dir}"/wandb/run-* "${result_dir}"/wandb/offline-run-* 2>/dev/null | head -n 1)
  if [ -n "${latest_wandb_run_dir}" ] && [ -d "${latest_wandb_run_dir}" ]; then
    resolved_wandb_resume_id="${latest_wandb_run_dir##*-}"
    echo "AUTO_RESUME_WANDB_ID is true. Reusing wandb run id: ${resolved_wandb_resume_id}"
  else
    echo "AUTO_RESUME_WANDB_ID is true, but no previous wandb run dir found under ${result_dir}/wandb"
  fi
fi

if [ -n "${resolved_wandb_resume_id}" ]; then
  wandb_resume_arg="resume=${resolved_wandb_resume_id}"
  echo "wandb resume enabled with id: ${resolved_wandb_resume_id}"
else
  echo "wandb resume disabled."
fi

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
        $wandb_resume_arg \
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
        experiment.training.max_epochs=25 \
        experiment.training.batch_size=1 \
        experiment.training.optim.accumulate_grad_batches=20 \
        algorithm.backbone.use_checkpointing=[false,false,true,true] \
        ++experiment.training.checkpointing.save_top_k=1 \
        ++experiment.training.checkpointing.save_last=true \
        ++experiment.training.checkpointing.monitor=validation/loss \
        ++experiment.training.checkpointing.mode=min \
        ++experiment.training.checkpointing.save_weights_only=false \
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
ln -sfn ${latest_ckpt_abs} ${lasted_ckpt_path}

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