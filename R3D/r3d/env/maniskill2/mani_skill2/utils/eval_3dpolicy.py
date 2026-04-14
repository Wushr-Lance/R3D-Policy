import gymnasium as gym
import mani_skill2.envs
import numpy as np
import os
import torch
import pytorch3d.ops as torch3d_ops
import subprocess
'''
pickcube info : Info: {'elapsed_steps': 200, 'is_obj_placed': False, 'is_robot_static': False, 'success': False}
stackcube info :Info: {'elapsed_steps': 200, 'is_cubaA_grasped': False, 'is_cubeA_on_cubeB': False, 'is_cubeA_static': True, 'success': False}
peginsertion info: Info: {'elapsed_steps': 200, 'success': False, 'peg_head_pos_at_hole': array([-0.26642743,  0.03912554, -0.08389243], dtype=float32)}
'''
class Maniskill2Env():
    def __init__(self, env_name: str , downsample_points: int, video_save_dir: str):
        # Always use offscreen rendering — we never need an interactive viewer
        # during training/eval. Also force the NVIDIA Vulkan ICD and trivial shader
        # to work on headless servers. This is harmless on desktop machines.
        os.environ.setdefault('VK_ICD_FILENAMES', '/usr/share/vulkan/icd.d/nvidia_icd.json')
        self.env = gym.make(
            env_name, obs_mode='pointcloud', control_mode='pd_joint_pos',
            shader_dir='trivial', renderer_kwargs={'offscreen_only': True}
        )
        self.env_name = env_name
        self.downsample_points = downsample_points
        self.success_num = 0
        self.eval_num = 0
        self.video_size = '128x128'
        self.video_save_dir = video_save_dir
        os.makedirs(self.video_save_dir, exist_ok=True)
        if self.env_name == 'PegInsertionSide-v0':
            self.pre_inserted_success =0
            self.is_grasped_success =0
    def __pad_array(self, numpy_array, target_num):
        current_num = numpy_array.shape[0]
        if current_num < target_num:
            print("Padding")
            if current_num == 0:
                numpy_array = np.zeros((target_num, 6))
            else:
                pad_num = target_num - current_num
                padding = np.zeros((pad_num, 6))
                numpy_array = np.vstack([numpy_array, padding])
        return numpy_array
    def fps(self, points, num_points=1024, use_cuda=True):
        K = [num_points]
        if points.shape[0]==0:
            return points, torch.tensor([])
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices
    def __downsample_pcd(self, pointcloud, downsample_points=None, use_cuda=True):
        '''
        pointcloud: (N, 6) numpy array
        downsample_points: int or None; if int, downsample to this number of points using FPS;  if None, do not downsample
        return: (M, 6) numpy array
        '''
        pcd = pointcloud
        if downsample_points is not None:
            pcd = self.__pad_array(pcd,downsample_points)
            pcd_points = pcd[:, :3]
            _, indices = self.fps(pcd_points, num_points=downsample_points, use_cuda=use_cuda)
            pcd = pcd[indices.cpu().numpy()[0]]
        return pcd
    def __extract_obs(self, obs):
        rgb = obs['image']['base_camera']['Color'][:, :, :3]*255.0  # (H, W, 3)
        self.ffmpeg.stdin.write(rgb.astype(np.uint8).tobytes())
        if self.env_name == 'PickCube-v0':
            goal_pose = obs['extra']['goal_pos']
            qpos = obs['agent']['qpos']
            state = np.concatenate([qpos, goal_pose], axis=-1)
            pointcloud_xyzw = obs['pointcloud']['xyzw']
            pointcloud_rgb = obs['pointcloud']['rgb']
            pointcloud = np.concatenate([pointcloud_xyzw[:, :3], pointcloud_rgb], axis=-1)
            pcd_seg = obs['pointcloud']['Segmentation']
            mask = pcd_seg[:, 0] != 12  # exclude Ground
            pointcloud = pointcloud[mask]
            pointcloud = self.__downsample_pcd(pointcloud, downsample_points=self.downsample_points, use_cuda=True)
        elif self.env_name == 'StackCube-v0':
            qpos = obs['agent']['qpos']
            state = qpos
            pointcloud_xyzw = obs['pointcloud']['xyzw']
            pointcloud_rgb = obs['pointcloud']['rgb']
            pointcloud = np.concatenate([pointcloud_xyzw[:, :3], pointcloud_rgb], axis=-1)
            pcd_seg = obs['pointcloud']['Segmentation']
            mask = pcd_seg[:, 0] != 12  # exclude Ground
            pointcloud = pointcloud[mask]
            pointcloud = self.__downsample_pcd(pointcloud, downsample_points=self.downsample_points, use_cuda=True)
        else:
            qpos = obs['agent']['qpos']
            state = qpos
            pointcloud_xyzw = obs['pointcloud']['xyzw']
            pointcloud_rgb = obs['pointcloud']['rgb']
            pointcloud = np.concatenate([pointcloud_xyzw[:, :3], pointcloud_rgb], axis=-1)
            pcd_seg = obs['pointcloud']['Segmentation']
            mask = pcd_seg[:, 0] != 13  # exclude Ground
            pointcloud = pointcloud[mask]
            pointcloud = self.__downsample_pcd(pointcloud, downsample_points=self.downsample_points, use_cuda=True)
        obs_dict = dict()
        obs_dict['state'] = state
        obs_dict['pointcloud'] = pointcloud
        return obs_dict
    


    def reset(self, seed):
        obs, _ = self.env.reset(seed=seed)
        self.state = True
        self.success = False
        self.count = 0
        if self.env_name == 'PegInsertionSide-v0':
            self.max_steps = 200
            self.pre_inserted = False
            self.is_grasped = False
        elif self.env_name == 'PickCube-v0':
            self.max_steps = 200
        elif self.env_name == 'StackCube-v0':
            self.max_steps = 200
        print(f'\rSteps: {self.count}/{self.max_steps}', end='', flush=True)

        video_dir = os.path.join(self.video_save_dir, f"{self.env_name}_seed{seed}")
        self.ffmpeg = subprocess.Popen([
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-pixel_format', 'rgb24',
            '-video_size', self.video_size,
            '-framerate', '10',
            '-i', '-',
            '-pix_fmt', 'yuv420p',
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-tune', 'zerolatency',
            '-g', '15',
            '-threads', '0',
            f'{video_dir}.mp4'
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        obs = self.__extract_obs(obs)
        return obs
    def __close_ffmpeg(self):
        if self.ffmpeg:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            self.ffmpeg = None
    def __update_state(self, terminated, truncated):
        if self.success:
            self.state = False
            self.success_num +=1
            self.eval_num +=1
            self.__close_ffmpeg()
            if self.env_name == 'PegInsertionSide-v0':
                if self.pre_inserted:
                    self.pre_inserted_success +=1
                if self.is_grasped:
                    self.is_grasped_success +=1
            print()
            print(f"Scuccess! Current Success Rate: {self.success_num}/{self.eval_num}")
            if self.env_name == 'PegInsertionSide-v0':
                print(f"Pre-inserted Success Rate: {self.pre_inserted_success}/{self.eval_num}, Is-grasped Success Rate: {self.is_grasped_success}/{self.eval_num}")
            return
        if terminated or truncated:
            self.state = False
            self.eval_num +=1
            self.__close_ffmpeg()
            if self.env_name == 'PegInsertionSide-v0':
                if self.pre_inserted:
                    self.pre_inserted_success +=1
                if self.is_grasped:
                    self.is_grasped_success +=1
            print()
            print(f"Failed! Current Success Rate: {self.success_num}/{self.eval_num}")
            if self.env_name == 'PegInsertionSide-v0':
                print(f"Pre-inserted Success Rate: {self.pre_inserted_success}/{self.eval_num}, Is-grasped Success Rate: {self.is_grasped_success}/{self.eval_num}")
            return

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self.count +=1
        print(f"\r \033[32mSteps: {self.count}/{self.max_steps}\033[0m", end='', flush=True)
        obs = self.__extract_obs(obs)
        self.success = info['success']
        if self.env_name == 'PegInsertionSide-v0':
            self.pre_inserted = info['pre_inserted'] if not self.pre_inserted else self.pre_inserted
            self.is_grasped = info['is_grasped'] if not self.is_grasped else self.is_grasped
        self.__update_state(terminated, truncated)
        
        return obs, self.state, self.success

    def reset_stats(self):
        """Reset per-run success counters without tearing down the underlying
        SAPIEN scene. Called at the start of each evaluation round so that the
        env instance can be reused across multiple calls to the runner."""
        self.success_num = 0
        self.eval_num = 0
        if self.env_name == 'PegInsertionSide-v0':
            self.pre_inserted_success = 0
            self.is_grasped_success = 0

    def get_stats(self):
        """Return current-run stats and print a summary, without closing the
        underlying gym env. Use this between eval rounds."""
        print()
        print(f"\033[31mFinal Success Rate: {self.success_num}/{self.eval_num}\033[0m")
        if self.env_name == 'PegInsertionSide-v0':
            print(f"Final Pre-inserted Success Rate: {self.pre_inserted_success}/{self.eval_num}, Final Is-grasped Success Rate: {self.is_grasped_success}/{self.eval_num}")
        return self.success_num, self.eval_num

    def close(self):
        self.env.close()
        print()
        print(f"\033[31mFinal Success Rate: {self.success_num}/{self.eval_num}\033[0m")
        if self.env_name == 'PegInsertionSide-v0':
            print(f"Final Pre-inserted Success Rate: {self.pre_inserted_success}/{self.eval_num}, Final Is-grasped Success Rate: {self.is_grasped_success}/{self.eval_num}")
        return self.success_num, self.eval_num

    



# env = gym.make("PickCube-v0", obs_mode="pointcloud", control_mode="pd_joint_pos", render_mode="human")
# print("Observation space", env.observation_space)
# print("Action space", env.action_space)

# obs, _ = env.reset(seed=0) # reset with a seed for randomness
# terminated, truncated = False, False
# while not terminated and not truncated:
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)
#     env.render()  # a display is required to render
# env.close()