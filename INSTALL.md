# Building Conda Environment from Zero to Hero

1.Create a conda env

You can use a conda environment YML file to create the env:

```bash
conda env create -f conda_environment.yml
conda activate r3d
```

Or create a conda env manually:

```bash
conda create -n r3d python=3.10 -y
conda activate r3d

# Install additional dependencies
pip install -r requirements.txt
```

---

2.Install CuRobo

```bash
cd R3D/r3d/env/robotwin2/envs
git clone https://github.com/NVlabs/curobo.git
cd curobo
pip install -e . --no-build-isolation
cd ../../../../../..
```

---

3.install torch

```bash
# make sure using cuda>=12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

4.install R3D

```bash
cd R3D
pip install -e . && cd ..
```

---

5.install pytorch3d

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

---

6.adjust code in `mplib`

This is following [RoboTwin Installation](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#3-basic-env:~:text=the%20mplib%20installed.-,Remove,-or%20collide)

> You can use `pip show mplib` to find where the `mplib` installed.

```
# mplib.planner (mplib/planner.py) line 807
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```

---

<!-- 7.install ManiSkill

Because ManiSkill and RoboTwin use different versions of Sapien, running results on the ManiSkill requires a separate conda environment.

```bash
cd R3D/r3d/env/maniskill2
pip install -e .
```

--- -->