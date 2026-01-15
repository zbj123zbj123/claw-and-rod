# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        #机器人特有的缓冲/视图初始化：例如获取并保存 Isaac Gym 的 Tensor 视图：self.root_states（形状一般 [num_actors_total, 13]）self.dof_state（[num_dofs_total, 2]，pos/vel）
#self.contact_forces、self.rigid_body_state、命令/噪声缓存、历史动作等
#建立索引映射（某 env 的第 j 个关节/actor 在全局张量里的行号）
#预分配 self.torques 等控制输出张量
#这些是 BaseTask 不知道的“业务细节”，由子类完成。
        self._prepare_reward_function()
        #奖励函数注册/权重表：遍历配置里开启的奖励/惩罚项，建立一个“可调用的列表或表格”，post_physics_step() 调用时自动按权重求和。
        self.init_done = True
#标记初始化完成。很多代价项会在 init_done=False 期间跳过（避免“出生瞬间”的非物理惩罚）。
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
#做什么：从配置里取出动作裁剪阈值，通常是一个标量（也可支持向量），表示动作允许的绝对值上限。
#来自哪里：LeggedRobotCfg.normalization.clip_actions。它独立于“动作缩放”参数（如 control.action_scale 或 PD 的 Kp/Kd），作用是先把动作限制到安全范围，再进入力矩/目标计算。
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        if self.tune_on:
            print("actions: ", self.actions)
        #语义：对 actions 逐元素裁剪到区间 [-clip_actions, clip_actions]。把裁剪后的张量放到 环境张量所在设备（
#等价函数：torch.clamp(...)。
        # step physics and render each frame
        self.render()
        #在每个控制周期开头调用一次 render()：处理 Viewer 事件（Esc 退出、V 切限速）；若仿真在 GPU，则 fetch_results 上一帧结果，保证画的是最新状态；
#同步/绘制一帧或仅轮询事件（取决于 enable_viewer_sync）。
        for _ in range(self.cfg.control.decimation):
            #控制指令的 “更新频率” 是低频的
#控制的核心是 “关节该怎么动” 的指令（即 self.actions）同一份动作 self.actions，连续执行 decimation 次物理步。
#公式：
#物理步长 Δt_phys = sim_params.dt（例如 1/400 s）
#控制周期 Δt_ctrl = decimation * Δt_phys（例如 decimation=4 ⇒ 1/100 s）
#这样可用高频物理保证数值稳定，同时保持较低的控制频率。
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            power2 = (self.torques[:, 1] * self.dof_vel[:, 1])
            # print("mean_power2:", power2.mean().item(),
            #       "neg_ratio:", (power2 < 0).float().mean().item())

            #print("torques: ", self.torques)
            #做什么：把无量纲动作转换为可下发的关节力矩（或等价驱动力）。典型：
#PD 位置控制：τ = Kp (q* − q) + Kd (qd* − qd)，动作常代表目标位姿增量或归一化角度；
#速度控制：τ = Kd (qd* − qd)；
#直接力矩控制：τ = action * τ_max（再限幅）。
#.view(self.torques.shape)：把 _compute_torques 的输出重构成和 self.torques 完全相同的形状（通常是 [num_envs, dofs_per_env]）。
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            #把 所有 DOF（degree of freedom）= 关节自由度 的当前力矩 一次性传给仿真（PhysX）.
            #DOF：PD/速度控制要用到最新 q, qd，所以每个子步都要刷新；一个关节一个dof，q：joint position（关节位置）qd（qdot）joint elocity（关节速度）
#Root / Contact：多数算法只在控制周期末用一次（算观测/奖励/终止），放在 post_physics_step() 开头统一刷新，能减少 GPU↔CPU/设备间同步成本；
#Rigid body：只有当你的观测/奖励真的用到每个刚体的姿态（比如足端世界坐标）才刷新。
#gymtorch.unwrap_tensor(...)：把 PyTorch 张量桥接为底层能读的指针/缓冲；要求张量在正确设备上（GPU pipeline 时在 CUDA）。
            self.gym.simulate(self.sim)
            #做什么：推进一次物理子步，步长 Δt_phys。GPU 管线（use_gpu_pipeline=True）下，这通常是异步入队，GPU 管线里常把同步（fetch_results）推迟到统一的地方（比如 render()），减少频繁“等一下”的开销；CPU 仿真则在 fetch_results 时同步。
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
                #阻塞（wait=True）：像 input() 或 time.sleep() 一样，等待完成 → 再继续。优点是拿到的数据是确定的“最新结果”。
#非阻塞（wait=False）：立刻返回（通常返回一个是否完成的标志），你可以做别的事或稍后再查；但如果马上去读状态，可能还是上一次的旧数据（或触发隐式同步）。
            self.gym.refresh_dof_state_tensor(self.sim)
            #刷新 DOF 的 pos/vel 视图，使得下一次循环/后续计算能读到最新关节状态。
#为什么每步都 refresh？
#因为你需要基于最新状态计算下一物理子步的力矩（例如 PD 控制用到当前 q/qd）。
        self.post_physics_step()
#这是本环境的“主逻辑”函数，一般做：
#更新根/刚体状态视图（refresh_rigid_body_state_tensor 等）；

#计算观测 self.obs_buf、（可选）特权观测 self.privileged_obs_buf；

#逐项计算奖励/惩罚并汇总到 self.rew_buf；

#判终止（摔倒、越界、时间到）→ 更新 self.reset_buf / self.time_out_buf；

#维护 self.extras：把奖励分项、诊断指标填进去（给 logger）。
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        #从观测里取出观测裁减阈值
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        #类比上面action动作，把普通观测逐元素裁剪到区间 ，把裁剪后的张量放到环境张量所在设备
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
#返回五件套（PPO runner 期望的统一接口）：
#obs_buf（裁剪后）
#privileged_obs_buf（可能为 None）
#rew_buf
#reset_buf（哪些 env 下步要重置）
#extras（日志/诊断，不参与训练）
    #step总：读动作 → 裁剪、搬到正确设备
#渲染事件（可选，同步/限速切换）

#decimation 循环（每次都是“算力矩→下发→物理一步→读回 DOF”）

#post_physics_step（算观测、奖励、终止、extras）

#裁剪观测并返回（obs, p_obs, rew, done, info）
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for  computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        #每个控制周期后刷；用来得到基座位姿/速度
        self.gym.refresh_net_contact_force_tensor(self.sim)
        #控足端接触/滑移等奖励/终止时刷，刚体合力接触张量从引擎写回到你的 torch 视图


        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
       # self.base_quat[:] = self.root_states[:, 3:7]
        actors_per_env = 2  # 你现在每个环境里有 2 个 actor：机器人 + 物体
        root_states = self.root_states.view(self.num_envs, actors_per_env, 13)
        self.base_quat = root_states[:, 0, 3:7]
        #base_quat：基座朝向四元数（xyzw），来源于根状态列 3:7
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, root_states[:, 0,7:10])
        #quat_rotate_inverse(q, v)：把世界系向量 v旋转到机体坐标系
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, root_states[:, 0,10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
#base_lin_vel：机体系线速度（policy 常用）
#base_ang_vel：机体系角速度（policy 常用）
#projected_gravity：重力向量在机体系的投影（非常常见的观测，能提供姿态信息，等价 IMU 加速度在静止时的读数）
        self._post_physics_step_callback()
#这段代码是 LeggedGym 中 _post_physics_step_callback 方法的实现，它是物理仿真步之后、计算奖励 / 终止条件 / 观测之前的关键回调函数，
 # 主要负责环境动态更新（如指令重采样、地形高度测量、随机推机器人等）。
        self.check_termination()
        #判终止 check_termination()：根据摔倒、越界、接触异常、时间上限（看 episode_length_buf）等设置

#reset_buf[i]=1（要重置）

#time_out_buf[i]=True/False（是否“到时”导致的 done）
        self.compute_reward()
        #算奖励 compute_reward()：把奖励分项（跟踪速度、能耗惩罚、足端滑移惩罚、平衡等）按权重加总到 rew_buf，并可把分项写到 extras 里以便记录。
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        #挑出要重置的 env
        self.reset_idx(env_ids)
        # # 重置终止的环境（如机器人位置、速度等）
        #self._reset_envs(env_ids)
        # 为重置后的环境重新采样新指令（核心调用）
        #self._resample_commands(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
#重新产出观测 self.compute_observations()：把最新状态组装成 obs_buf（以及 privileged_obs_buf）。
#注释里提醒：有的观测（例如某些体位点）可能需要你先 refresh_*_tensor 或额外模拟一步才能更新完全，这取决于你的实现。
        #self.prev_rel_height[:] = self._relative_base_height().detach()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 0,7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # root = self.root_states.view(-1, 13)
        # quat = root[self.robot_actor_indices, 3:7]  # [N,4] (x,y,z,w)
        #
        # # 计算 base 的“上方向”在世界系里指向哪里
        # z_axis = torch.zeros((self.num_envs, 3), device=self.device)
        # z_axis[:, 2] = 1.0
        # up = quat_rotate(quat, z_axis)
        # h = root[self.robot_actor_indices, 2]
        # upz = up[:, 2]
        # flipped = (upz < -0.5)
        # self.h_peak = torch.maximum(self.h_peak, h)
        # self.has_reached_high = self.h_peak >= 2.18
        # self.reset_buf |=(self.has_reached_high & flipped)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
          # 应是 0~49
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)


        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        #self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf




    #选择性重置部分环境实例（通过环境ID指定），是强化学习中多环境并行训练时的关键方法。它不仅会重置环境状态，还会处理课程学习（curriculum）、缓冲区清零和日志记录等逻辑。
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            #self.reward_functions[i]() 返回的形状为 [4]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            # 累加至该奖励在当前回合的总和
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
            #若配置了 only_positive_rewards=True，则通过 torch.clip 将总奖励截断为非负值（即所有负奖励都变为 0）。
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            #_reward_termination()：专门计算回合终止时的额外奖励（例如成功到达目标时的大额奖励 +100，或失败时的惩罚 -50）。
    #逻辑：终止奖励通常在其他奖励计算和截断之后添加，确保其不受 “仅保留正奖励” 的影响（例如即使总奖励被截断为 0，终止奖励仍能正常生效）。
#在最后进行终止奖励计算，因为前面的REWARDFUNCTION里面没有他，CONTINUE了之前的PREPARE REWARD
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind让智能体在决策时感知到当前的任务目标（如 “需要以 0.5m/s 的速度前进”），从而学习如何调整动作来匹配指令。
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_limits_hard = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                low = props["lower"][i].item()
                high = props["upper"][i].item()
                self.dof_pos_limits_hard[i, 0] = low
                self.dof_pos_limits_hard[i, 1] = high

            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props
#“环境创建阶段”被调用，用来读取并缓存 URDF 中的关节（DOF）限制与能力参数（速度/力矩），并在本地保存“软限制”。“环境创建阶段”被调用，
    # 用来读取并缓存 URDF 中的关节（DOF）限制与能力参数（速度/力矩），并在本地保存“软限制”。在奖励reward_dof_pos_limits
    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        #它是物理仿真步之后、计算奖励 / 终止条件 / 观测之前的关键回调函数，
        # 主要负责环境动态更新（如指令重采样、地形高度测量、随机推机器人等）。
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """

        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
       # 在奖励计算中被使用（评估任务完成度）,在观测计算中被使用（让智能体 “看到” 任务目标）这两个是self.commands的作用
        #resample的调用：1.在环境step方法中，检测到终止环境后调用当智能体与环境交互一步（step）并检测到某些环境终止（dones中有True）时，会先重置这些环境，再为它们重新采样指令
        # 在环境 reset 方法中，初始化所有环境时调用当首次创建环境或手动重置所有环境时，需要为所有环境生成初始指令：set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        root = self.root_states.view(-1, 13)
        h = root[self.robot_actor_indices, 2]

        enter_hold = (h > 1.85)
        exit_hold = (h < 1.70)
        self.is_hold = (self.is_hold | enter_hold) & (~exit_hold)
        self.action_scale_vec = torch.ones(self.num_envs,self.num_dof, device=self.device) * self.cfg.control.action_scale
        self.action_scale_vec[self.is_hold,:]=0.05
        # left_idx=self.dof_names.index('P_2_to_left_Link')
        # right_idx=self.dof_names.index('P_2_to_right_Link')
        # self.action_scale_vec[left_idx] = 0.18
        # self.action_scale_vec[right_idx] = 0.20
        actions_scaled = actions * self.action_scale_vec
        # self.claw_idx =[left_idx, right_idx]
        # actions_scaled[:,self.claw_idx] = torch.tensor(0, device=actions_scaled.device, dtype=actions_scaled.dtype)

        if self.tune_on:
            qd_ref = torch.zeros_like(self.dof_vel)
            # qd_scalar = self.tune_A * (2.0 * math.pi * self.tune_f) * math.cos(
            #     2.0 * math.pi * self.tune_f * self.tune_t)
            # qd_ref[:, self.tune_jid] = torch.tensor(qd_scalar, device=self.dof_vel.device, dtype=self.dof_vel.dtype)
            self.tune_t += self.tune_dt
            # s = math.sin(2 * math.pi * self.tune_f * self.tune_t)
            # offset = max(-self.cfg.control.action_scale, min(self.tune_A * s, self.cfg.control.action_scale))
            offset=self.tune_A if self.tune_t >= self.tune_delay else 0.0
            actions_scaled.zero_()
            actions_scaled[:,self.tune_jid]=torch.tensor(offset, device=actions_scaled.device, dtype=actions_scaled.dtype)
            pos_target = actions_scaled + self.default_dof_pos
            tgt0 = float(pos_target[0, self.tune_jid].item())
            pos0 = float(self.dof_pos[0, self.tune_jid].item())
            e0 = tgt0 - pos0
            self.tr_t.append(self.tune_t)
            self.tr_ref.append(float(pos_target[0,self.tune_jid].item()))
            self.tr_pos.append(pos0)
            self.tr_err.append(e0)
            self.err_win.append(e0)
            if len(self.tr_t) % 200 == 0 and len(self.err_win) > 0:
                import numpy as np
                rms = float(np.sqrt(np.mean(np.square(list(self.err_win)))))
                step = len(self.tr_t)
                # Kp/Kd 兼容 [env,dof] 或 [dof]
                kp = float(self.p_gains[0, self.tune_jid].item()) if self.p_gains.dim() == 2 else float(self.p_gains[self.tune_jid].item())
                kd = float(self.d_gains[0, self.tune_jid].item()) if self.d_gains.dim() == 2 else float(self.d_gains[self.tune_jid].item())
                self.tb.add_scalar("tune/rms_env0_" + self.tune_joint, rms, step)
                self.tb.add_scalar("tune/Kp_" + self.tune_joint, kp, step)
                self.tb.add_scalar("tune/Kd_" + self.tune_joint, kd, step)

                # 可选：顺带把两张图塞进 TensorBoard（也会保存到文件夹）
                # 目标 vs 实际
                #方法中使用的变量（如 plt），会在 “方法定义的模块” 中查找，而不是 “方法调用的子类” 中查找
                import matplotlib
                matplotlib.use("Agg")  # 服务器无显示也能出图
                import matplotlib.pyplot as plt
                fig1 = plt.figure(figsize=(7, 4))
                plt.plot(self.tr_t, self.tr_ref, label="ref")
                plt.plot(self.tr_t, self.tr_pos, label="pos")
                plt.xlabel("time (s)");
                plt.ylabel("rad");
                plt.title(self.tune_joint + " ref vs pos");
                plt.legend();
                plt.grid(True, alpha=0.3)
                self.tb.add_figure("tune/pos_vs_ref_" + self.tune_joint, fig1, global_step=step)
                fig1.savefig(f"/home/abc/isaacgym/legged_gym/logs/tb_pid_tune/pos_vs_ref_{self.tune_joint}.png",
                             dpi=150)
                plt.close(fig1)

                # 误差
                fig2 = plt.figure(figsize=(7, 4))
                plt.plot(self.tr_t, self.tr_err, label="error")
                plt.xlabel("time (s)");
                plt.ylabel("rad");
                plt.title(self.tune_joint + " error");
                plt.legend();
                plt.grid(True, alpha=0.3)
                self.tb.add_figure("tune/error_" + self.tune_joint, fig2, global_step=step)
                fig2.savefig(f"/home/abc/isaacgym/legged_gym/logs/tb_pid_tune/error_{self.tune_joint}.png", dpi=150)
                plt.close(fig2)

                t_np = np.array(self.tr_t, dtype=np.float64)
                ref_np = np.array(self.tr_ref, dtype=np.float64)
                pos_np = np.array(self.tr_pos, dtype=np.float64)
                percent_band1 = 0.02
                metrics = self.compute_step_metrics(t=t_np, ref=ref_np, pos=pos_np, percent_band=percent_band1)
                if metrics is not None:
                    Mp = metrics["Mp"]
                    tr = metrics["tr"]
                    ts = metrics["ts"]
                    ess = metrics["ess"]
                    rms = metrics["rms"]

                    step = len(self.tr_t)
                    kp = float(self.p_gains[0, self.tune_jid].item()) if self.p_gains.dim() == 2 else float(
                        self.p_gains[self.tune_jid].item())
                    kd = float(self.d_gains[0, self.tune_jid].item()) if self.d_gains.dim() == 2 else float(
                        self.d_gains[self.tune_jid].item())

                    # 标量写入 TensorBoard
                    self.tb.add_scalar("tune/rms_env0_" + self.tune_joint, rms, step)
                    self.tb.add_scalar("tune/Mp_" + self.tune_joint, Mp, step)
                    if tr is not None:
                        self.tb.add_scalar("tune/tr_" + self.tune_joint, tr, step)
                    if ts is not None:
                        self.tb.add_scalar("tune/ts_" + self.tune_joint, ts, step)
                    self.tb.add_scalar("tune/ess_" + self.tune_joint, ess, step)
                    self.tb.add_scalar("tune/Kp_" + self.tune_joint, kp, step)
                    self.tb.add_scalar("tune/Kd_" + self.tune_joint, kd, step)

                    # 终端也打印一份，方便你看
                    # print(f"[tune-{self.tune_joint}] step={step}  "
                    #       f"rms={rms:.4f} rad  Mp={Mp:.1f}%  tr={tr:.3f}s  ts={ts:.3f}s  ess={ess:.4f} rad  "
                    #       f"Kp={kp:.1f}  Kd={kd:.1f}")
        self.tune_k += 1
        control_type = self.cfg.control.control_type
        # print('actions_scaled', actions_scaled.shape)
        # print('default_dof_pos', self.default_dof_pos.shape)
        # print('dof_pos', self.dof_pos.shape)
        # print('dof_vel', self.dof_vel.shape)
        # print('p_gains', self.p_gains.shape)
        # print('d_gains', self.d_gains.shape)

        if control_type=="P":
            # err = (actions_scaled + self.default_dof_pos - self.dof_pos)

            # 目标角
            q_des = actions_scaled + self.default_dof_pos  # [num_envs, num_dof]
            lower = self.dof_pos_limits_hard[:, 0]  # [num_dof]
            upper = self.dof_pos_limits_hard[:, 1]  # [num_dof]
            q_des = torch.clamp(q_des, lower, upper)


            kd2_swing = 4.0
            kd2_hold = 25.0

            d_eff = self.d_gains.unsqueeze(0).expand(self.num_envs, -1).clone()
            d_eff[:, 1] = kd2_swing
            d_eff[self.is_hold, 1] = kd2_hold
            torques = self.p_gains*(q_des - self.dof_pos) - d_eff*self.dof_vel

            # print("err first2:", err[0, :2].tolist(),
            #       "Kp:", self.p_gains[:2].tolist(),
            #       "raw:", torques[0, :2].tolist(),)
            if self.tune_on:
                torques = self.p_gains * (pos_target - self.dof_pos) - self.d_gains * (self.dof_vel - qd_ref)  # 调参的时候用
            # print("sss",self.default_dof_pos-self.dof_pos)
            #print("ttt3",torques)
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)#torch.clip(torques,-0.0,0.0)torch.clip(torques, -self.torque_limits, self.torque_limits)#torch.clip(torques,-0.0,0.0)torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_step_metrics(self,t, ref, pos, percent_band):
        """
        t, ref, pos: 1D numpy 数组，长度相同
        percent_band: 整定带宽（默认 2%）
        返回一个字典: {'Mp': ..., 'tr': ..., 'ts': ..., 'ess': ..., 'rms': ...}
        """
        t = np.asarray(t)
        ref = np.asarray(ref)
        pos = np.asarray(pos)

        if len(t) < 5:
            return None  # 数据太少

        # 1) 找阶跃开始时刻（ref 第一次偏离初值）
        r0 = ref[0]
        idx_step = np.where(np.abs(ref - r0) > 1e-4)[0]
        if len(idx_step) == 0:
            return None  # 没找到阶跃
        i0 = idx_step[0]
        t0 = t[i0]

        # 2) 最终参考值（取最后 N 个点平均，更稳）
        N_tail = max(10, len(ref) // 10)
        r_inf = ref[-N_tail:].mean()
        A = r_inf - r0  # 阶跃幅值（可能为负）
        if np.abs(A) < 1e-4:
            return None

        # 3) 归一化一个方向（考虑正/负阶跃）
        sgn = np.sign(A)
        y = sgn * pos  # 归一化后相当于正阶跃
        y0 = sgn * r0
        y_inf = sgn * r_inf
        amp = np.abs(A)

        # 4) 超调量 Mp（%）
        peak = y[i0:].max()
        Mp = max(0.0, (peak - y_inf) / amp) * 100.0  # 超调百分比

        # 5) 上升时间 tr：从 10% 到 90%
        y10 = y0 + 0.1 * amp
        y90 = y0 + 0.9 * amp

        def _first_cross(y_arr, thr):
            idx = np.where(y_arr >= thr)[0]
            return None if len(idx) == 0 else idx[0]

        i10 = _first_cross(y[i0:], y10)
        i90 = _first_cross(y[i0:], y90)
        tr = None
        if i10 is not None and i90 is not None and i90 > i10:
            tr = t[i0 + i90] - t[i0 + i10]

        # 6) 整定时间 ts：进入 ±percent_band 并保持
        band = percent_band * amp
        e = pos - r_inf
        # 找最后一次出带的位置
        out_band = np.where(np.abs(e) > band)[0]
        ts = None
        if len(out_band) > 0:
            last_out = out_band[-1]
            if last_out > i0:
                ts = t[last_out] - t0
            else:
                ts = 0.0

        # 7) 稳态误差 ess：末尾一段的平均误差
        ess = e[-N_tail:].mean()

        # 8) RMS 误差（从阶跃开始算）
        rms = float(np.sqrt(np.mean((ref[i0:] - pos[i0:]) ** 2)))

        return {
            "Mp": Mp,  # 超调百分比 %
            "tr": tr,  # 上升时间 (s)
            "ts": ts,  # 整定时间 (s)
            "ess": ess,  # 稳态误差 (rad)
            "rms": rms,  # RMS 误差 (rad)
        }

    def _reset_dofs(self, env_ids):#只有在done或者意外终止的时候才reset
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dof), device=self.device)
        #gai
        if self.tune_on:
            self.dof_pos[env_ids] = self.default_dof_pos
        # 打印形状验证
        self.dof_vel[env_ids] = 0.
        ds = self.dof_state.view(self.num_envs, self.num_dofs, 2)  # → [N, n_dof, 2]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(self.robot_actor_indices[env_ids].to(dtype=torch.int32)), len(env_ids_int32))

        # print("离开 _reset_dofs 时的",self.robot_actor_indices[env_ids])
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:#custom_origins 一般表示**你启用了“自定义环境原点/布局”**的模式：
#在这种模式下，多数任务希望进一步增加多样性/鲁棒性，让同一环境中出生点在中心附近小范围随机；
            self.root_states[env_ids] = self.base_init_state
            #create env里面
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
#先按 start_pose 生成，接着你可以用 root_state 覆盖到你想要的位置。
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 0,7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:48] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        #contact_forces 通常是一个4 维张量，形状为：(num_envs, num_bodies, num_contacts, 3)num_envs：仿真环境的数量（多环境并行时）；
    #num_bodies：机器人的刚体总数（如躯干、大腿、脚等）；
    #num_contacts：每个刚体可能同时存在的接触点数量（通常取最大值，如 4）；
    #3：接触力在 x、y、z 三个方向的分量（单位：牛顿，N）。
        #Isaac Gym 的 net_contact_forces 就是“净接触力”，已经把同一刚体上的多个接触点求和了，
        # 所以没有额外的“num_contacts”维度（不是 4 维）。

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        rb = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rb_t = gymtorch.wrap_tensor(rb)

        # 计算总刚体数并做一致性检查
        # rb_t 应该是 [total_bodies, 13]
        assert rb_t.numel() % 13 == 0, "Rigid body state tensor size not multiple of 13"
        total_bodies = rb_t.numel() // 13
        assert total_bodies % self.num_envs == 0, \
            f"total_bodies={total_bodies} 不能被 num_envs={self.num_envs} 整除"
        self.num_bodies = total_bodies // self.num_envs

        # 现在再 reshape
        self.rigid_body_states = rb_t.view(self.num_envs, self.num_bodies, 13)
        # create some wrapper tensors for different slices
        actors_per_env=2
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, actors_per_env, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # print("dofstate:", self.dof_state)
        # print(self.dof_state.shape[0])
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]#view 不是拷贝，是同一块内存的不同“看法”。所以你改 view，其实就是在改同一块底层内存，原张量当然也跟着变。
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 0,3:7]


        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs,-1,3)# shape: num_envs, num_bodies, xyz axis
        # print(self.contact_forces.shape)
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # 初始化一次（_init_buffers 里）
        self.last_h = torch.zeros(self.num_envs, device=self.device)

        self.is_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.p_gains_base = self.p_gains.clone()
        self.d_gains_base = self.d_gains.clone()
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:,0, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        #self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        #self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 0,10:13])
        actors_per_env = 2  # 你现在每个环境里有 2 个 actor：机器人 + 物体
        root_states = self.root_states
        self.base_quat = root_states[:, 0, 3:7]  # 只取第 0 个（机器人）
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            #倒数第二个方法points
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()#每个 AssetOptions 只在对应的 load_asset(...) 那一次生效在那个资产上，互不影响。不同资产当然可以、也应该用不同的选项,给了一个类
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        print("dof_names:", self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            #筛选出「碰撞时需要被惩罚」的部件名称。
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            #筛选出「碰撞时直接终止当前 episode（回合）」的部件名称
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        #这是一个自定义方法，通常用于计算多个仿真环境的原点位置。在 Isaac Gym 中，多环境（multi-environment）通常以网格形式排列，每个环境有独立的原点坐标
        env_lower = gymapi.Vec3(0., 0., 0.)
        #定义单个环境的边界（下界和上界）。这里都设为 (0,0,0) 可能是占位符，实际使用时会根据环境大小设置（如 env_lower = gymapi.Vec3(-2, -2, 0)，env_upper = gymapi.Vec3(2, 2, 0) 表示一个 4x4 的平面区域）。用于限制机器人在环境内活动，或在环境中随机生成物体。
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        #初始化一个列表，用于存储每个环境中机器人（actor）的句柄（handle）。句柄是 Isaac Gym 中标识仿真对象的唯一 ID，后续通过句柄可操作机器人（如获取状态、施加控制）。
        self.envs = []
        #初始化一个列表，用于存储所有仿真环境的句柄。Isaac Gym 中每个环境是独立的仿真实例，通过环境句柄可操作特定环境
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            #只是告诉 Gym “我不提供包围盒，你按我自己放好的位置来”。你的代码正是这种做法——用 self.env_origins[i] 把每个 env 的东西摆开，所以就算边界是 0，也不会重叠。
            #env 边界不是“墙”，不会阻止跨 env 碰撞。真正避免相互影响，要把不同 env 的 actor 放得足够远（你已经在做）。
            pos = self.env_origins[i].clone()#确定每个环境的原点，但是可能会有一样的，所以下面一行加了变化来保证每个环境原点不一样，但是视觉上可能会有重合，也仅限于地形模式（heightfield和trimesh）
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)#摩擦随机化每个环境的所有形状同一个摩擦系数，每个环境不同
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            print("i的handel：",actor_handle)
            #先按 start_pose 生成，接着你可以用 root_state 覆盖到你想要的位置。第 5 个参数 i：collision group（碰撞组号）。常见做法是用环境索引当组号，这样不同 env 之间默认不会互相碰撞
            # env_handle：要把机器人实例化到哪个 env。
            # robot_asset：已经用 gym.load_asset(...) 载入好的 资产（URDF/USDA 等）。
            # start_pose：初始位姿（gymapi.Transform，含 p 平移和 r 四元数）。决定 actor 出生时的世界坐标位置与朝向。
            # self.cfg.asset.name：这个 actor 的名字（调试、日志与可视化里用得到）。。
            dof_props = self._process_dof_props(dof_props_asset, i)
            #调用方法processdof props调整关节属性增加鲁棒性   读取为了预留可改动空间，给self.doflimit赋值，return的是dofprops
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            #获取当前机器人的刚体属性（如质量、惯性、阻尼等）。
            body_props = self._process_rigid_body_props(body_props, i)
            #自定义方法，可能根据环境索引 i 调整刚体属性（如随机化质量分布）。调用方法process rigid body props
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            #记录环境和机器人句柄

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
#为每只脚的刚体部件分配一个唯一索引，并存储在张量中。#self.feet_indices 是一个张量，存储所有脚部刚体的索引（如[5, 6, 7, 8]，假设 4 只脚对应索引 5-8）。后续可通过这些索引快速获取脚部的接触力、位置等信息（如计算脚部空中时间）。
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
#记录所有需要惩罚碰撞的刚体部件的索引（如躯干、手臂等）
        print("惩罚",self.penalised_contact_indices)
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
#记录所有碰撞后会触发 episode 终止的刚体部件的索引（如头部、颈部等）。
        self.feet_name_to_idx = {name: int(self.feet_indices[i].item())
                                 for i, name in enumerate(feet_names)}
        lf = self.feet_name_to_idx["LF_FOOT"]
        rf = self.feet_name_to_idx["RF_FOOT"]
        lh = self.feet_name_to_idx["LH_FOOT"]
        rh = self.feet_name_to_idx["RH_FOOT"]
        all_ids = torch.arange(len(body_names), device=self.device, dtype=torch.long)
        allowed_mask = torch.zeros_like(all_ids, dtype=torch.bool)
        feet_indices = list(self.feet_name_to_idx.values())
        allowed_mask[torch.tensor(feet_indices, device=self.device, dtype=torch.long)] = True
        self.allowed_contact_ids = torch.tensor(feet_indices, device=self.device, dtype=torch.long)  # 脚的索引 tensor
        self.forbidden_contact_ids = all_ids[~allowed_mask]
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False#env 的原点位置由地形决定（自定义 origins）
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt#4*0.005=0.02
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            #  物理仿真步之后的环境动态更新if self.cfg.terrain.measure_heights:
            #self.measured_heights = self._get_heights()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
#    if self.cfg.terrain.measure_heights:
#  self.height_points = self._init_height_points()在initbuffer里面

        points += self.terrain.cfg.border_size
        #self.terrain = Terrain(self.cfg.terrain, self.num_envs)在createsim里面
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return self.is_hold * torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    def _reward_action_rate(self):
        da = self.last_actions - self.actions
        rate = torch.sum(torch.square(da), dim=1)

        gate = self.is_hold.float()  # [num_envs]，hold才罚
        return rate * gate

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return self.is_hold*torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
