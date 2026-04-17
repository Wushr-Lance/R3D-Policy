# R3D: Revisiting 3D Policy Learning
<a href="https://r3d-policy.github.io"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/pdf/2604.15281"><strong>Paper</strong></a>
  |
  <a href="https://"><strong>Twitter</strong></a> | <a href="https://youtu.be/659pPYQWuqA"><strong>YouTube</strong></a>

  <a href="https://hongzhengdong.github.io/">Zhengdong Hong*</a>, 
  <a href="https://wushr-lance.github.io/">Shenrui Wu*</a>, 
  <a href="https:// /">Haozhe Cui*</a>, 
  <a href="https:// /">Boyi Zhao</a>, 
  <a href="https:// /">Ran Ji</a>, 
  <a href="https:// /">Yiyang He</a>, 
  <a href="https:// /">Hangxing Zhang</a>, 
  <a href="https:// /">Zundong Ke</a>, 
  <a href="https://wang59695487.github.io/">Jun Wang</a>, 
  <a href="http://www.cad.zju.edu.cn/home/gfzhang/">Guofeng Zhang</a>
  <a href="https://jiayuan-gu.github.io/">Jiayuan Gu†</a>


<div align="center">
  <img src="R3D.png" alt="R3D" width="100%">
</div>

---

## 📋 Overview

R3D diagnoses why scaling 3D policy learning fails and proposes a new 3D imitation learning policy architecture characterized by the following key components:

* **Robust Scaling with Layer Normalization**: Overcomes the "scaling paradox" by replacing Batch Normalization with Layer Normalization, enabling the stable training of high-capacity 3D encoders (e.g., Uni3D).
* **Comprehensive 3D Data Augmentation**: Prevents overfitting through a robust pipeline including FPS randomization, color jitter, additive noise, and random point dropout.
* **Spatially-Aware Decoding**: Employs cross-attention between action queries and dense geometric tokens to preserve high-resolution spatial information, ensuring precision during complex manipulation tasks.
* **Large-Scale 3D Pre-training**: Leverages rich geometric priors by utilizing encoders pre-trained on diverse 3D datasets, which accelerates convergence and enhances feature representation.
* **Multi-Objective Decoding**: Incorporates auxiliary tasks, such as end-effector pose prediction, to provide stronger proprioceptive grounding and stabilize the denoising process.

R3D significantly outperforms state-of-the-art baselines across both simulation and real-world benchmarks. For more technical details and experimental results, please refer to our <a href="https://r3d-policy.github.io">project page</a> and <a href="https://arxiv.org/pdf/2604.15281">paper</a>.

Please follow the detailed instructions below to reproduce our results on both simulation and real-world.

---

## 🛠️ Installation

Follow the detailed instructions in [INSTALL.md](INSTALL.md) to set up the environment and install dependencies.

---

## 📥 Download

1.Download assets

```bash
cd R3D/r3d/env/robotwin2
bash script/_download_assets.sh
cd ../../../..
```

2.Download pre-processed data

It might take some time, be patient!

```bash
hf download eddie-cui/r3d --repo-type dataset --local-dir ./R3D/data
```

3.Download pre-trained weights of vision encoder 

```bash
hf download eddie-cui/r3d-weights --local-dir ./R3D/pretrain_weight
```

Pre-processed data and pre-trained weights of vision encoder are available on <a href="https://huggingface.co/datasets/eddie-cui/r3d">data</a> and <a href="https://huggingface.co/datasets/eddie-cui/r3d">weights</a>.

---

## ⚙️ Configuration Guide

The following quick-reference table provides the key parameters within our configuration files to facilitate the reproduction of our ablation studies.

* {policy}.yaml represents policy-specific configurations, such as `R3D/r3d/config/r3d_robotwin2.yaml` and `R3D/r3d/config/r3d_maniskill.yaml`
* {task}.yaml represents task-specific configurations, such as `R3D/r3d/config/task/robotwin2_demo_task.yaml` and `R3D/r3d/config/task/maniskill_PickCube.yaml`

| Component | Config Path | Key Parameter | Default Value |
| :--- | :--- | :--- | :--- |
| **Decoder** | `{policy}.yaml` | `condition_type` | `one_way_transformer` |
| **Decoder Layers** | `{policy}.yaml` | `transformer_config.depth` | `4` |
| **Point Cloud Color** | `{policy}.yaml` | `use_pc_color` | `true` |
| **Encoder Type** | `{policy}.yaml` | `pointnet_type` | `uni3d` |
| **Pretrained Encoder** | `{policy}.yaml` | `pointcloud_encoder_cfg.use_pretrained_weights` | `true` |
| **Normalization Type** | `{policy}.yaml` | `pointcloud_encoder_cfg.normalization_type` | `layer_norm` |
| **Global vs. Dense Feature** | `{policy}.yaml` | `pointcloud_encoder_cfg.feature_mode` | `pointsam` |
| **FPS Randomization** | `{policy}.yaml` | `fps_random_config.use_random` | `true` |
| **EE Prediction** | `{policy}.yaml` | `use_target_ee` | `false` |
| **Pointcloud and State Noise** | `{policy}.yaml` | `data_augmentation.use_augmentation` | `true` |
| **Color Jitter** | `{policy}.yaml` | `data_augmentation.use_color_jitter` | `true` |
| **Action Shape** | `{task}.yaml` | `action.shape` | `[14] or [8]` |
| **Evaluation Rollouts** | `{task}.yaml` | `env_runner.eval_episodes` | `100` |


---

## 🚀 Training & Evaluation

### Training Examples

1.RoboTwin
```bash
conda activate r3d && bash scripts/train_robotwin2_single.sh r3d_robotwin2 place_shoe 0000 0 0
# To enable DDP multi-GPU training, you only need to change GPU ID from single like "0" to multiple like "0,1" or "0,1,2"
conda activate r3d && bash scripts/train_robotwin2_single.sh r3d_robotwin2 place_shoe 0000 0 0,1
```

2.ManiSkill
```bash
conda activate r3d_maniskill && bash scripts/train_maniskill_single.sh r3d_maniskill PickCube 0000 0 0
# To enable DDP multi-GPU training, you only need to change GPU ID from single like "0" to multiple like "0,1" or "0,1,2"
conda activate r3d_maniskill && bash scripts/train_maniskill_single.sh r3d_maniskill PickCube 0000 0 0,1
```

---


## 🤖 Real-World Deployment

Coming soon!

---

## 🙏 Acknowledgements

Code borrows heavily from [3D-Diffusion-Policy](https://github.com/YanjieZe/3D-Diffusion-Policy).