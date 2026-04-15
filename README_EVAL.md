# Evaluation for Revisit Error and Reprojection Error

## Revisit Error
To run the revisit error evaluation, use the following command:
```bash
conda activate geometryforcing
pip install torch numpy tqdm opencv-python lpips

bash scripts/eval_rve.sh
```

## Reprojection Error

### Environment Setup
For the reprojection error, we recommend using a new conda environment to avoid dependency conflicts. You can create and activate the environment using the following commands:

```bash
# create and activate conda environment
conda create -n rpe_evaluation python=3.10 && conda activate rpe_evaluation

# install necessary packages
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install suitesparse -c conda-forge
pip install -U pip setuptools wheel
pip install open3d tensorboard scipy opencv-python tqdm matplotlib pyyaml opencv-python decord imageio pathlib
pip install "imageio[ffmpeg]" imageio-ffmpeg
pip install evo --upgrade --no-binary evo
pip install gdown

# optional alternative backend for mp4 writing (choose one backend is enough)
# pip install imageio[pyav]

# update submodules
git submodule update --init --recursive

# ensure lietorch Eigen headers are present
rm -rf evaluation/reprojection_error/third_party/DROID-SLAM/thirdparty/lietorch/eigen
git clone --depth 1 https://gitlab.com/libeigen/eigen.git evaluation/reprojection_error/third_party/DROID-SLAM/thirdparty/lietorch/eigen

# install third party libraries
pip install --no-build-isolation evaluation/reprojection_error/third_party/DROID-SLAM/thirdparty/lietorch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# install droid-backend
cd evaluation/reprojection_error/third_party/DROID-SLAM && pip install --no-build-isolation -e . && cd ../../../../..

# download DROID-SLAM checkpoint and place it in expected path
cd evaluation/reprojection_error/third_party/DROID-SLAM && bash ./tools/download_model.sh && cd ../../../../..
mkdir -p evaluation/reprojection_error/checkpoints
cp -f evaluation/reprojection_error/third_party/DROID-SLAM/droid.pth evaluation/reprojection_error/checkpoints/droid.pth
```

### Run the evaluation
To run the reprojection error evaluation, use the following command:
```bash
bash scripts/eval_rpe.sh
```