from time import time
import numpy as np
import os
from legged_gym.utils.terrain import Terrain
import torch
from torch import Tensor
from typing import Tuple, Dict
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from isaacgym import gymtorch, gymapi, gymutil
from .claw4_config import Claw4Cfg

class Claw4(LeggedRobot):
    cfg: Claw4Cfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.tune_on=False
        self.tune_joint="P_base_to_1_Link"
        self.tune_A=1.5
        self.tune_delay=1.0
        self.tune_f=0.50
        self.tune_dt=self.sim_params.dt
        self.tune_t=0.0
        names = self.dof_names
        self.tune_jid = names.index(self.tune_joint)
        self.tune_last_target = self.default_dof_pos.detach().clone() # 保存上一拍的 pos_target（全关节）
        self.tune_k = 0
        from collections import deque
        self.err_win = deque(maxlen=int(2.0 / self.tune_dt))  # 最近 2 秒的误差窗口
        from torch.utils.tensorboard import SummaryWriter
        import matplotlib
        matplotlib.use("Agg")  # 服务器无显示也能出图
        import matplotlib.pyplot as plt
        self.tb = SummaryWriter("/home/abc/isaacgym/legged_gym/logs/tb_pid_tune")
        self.tr_t, self.tr_ref, self.tr_pos, self.tr_err = [], [], [], []
        self.rod_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.rod_success_ema = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.stand_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.max_stand_hold_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.success_hold_time = 0.5
        self.success_hold_steps = int(self.success_hold_time / self.dt)
        # 每个env每个level一个actor index
        self.active_rod_level = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)
    def _create_envs(self):
        self.num_rod_levels = len(self.cfg.obj.files)
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()  # 每个 AssetOptions 只在对应的 load_asset(...) 那一次生效在那个资产上，互不影响。不同资产当然可以、也应该用不同的选项,给了一个类
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 100000  # 可调
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

        rod_assets = []
        for rod_path_tmpl in self.cfg.obj.files:
            rod_asset_path = rod_path_tmpl.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            rod_asset_root = os.path.dirname(rod_asset_path)
            rod_asset_file = os.path.basename(rod_asset_path)

            rod_options = gymapi.AssetOptions()
            rod_options.fix_base_link = True
            rod_options.disable_gravity = True
            rod_options.replace_cylinder_with_capsule = True

            rod_asset = self.gym.load_asset(self.sim, rod_asset_root, rod_asset_file, rod_options)
            rod_assets.append(rod_asset)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        rod_shape_props_asset = self.gym.get_asset_rigid_shape_properties(rod_asset)

        body_names1 = self.gym.get_asset_rigid_body_names(robot_asset)
        #print("body_names1:", body_names1)
        body_names2 = self.gym.get_asset_rigid_body_names(rod_asset)
        #print("body_names2:", body_names2)
        body_names = body_names1 + body_names2
        #print("body_names3:", type(body_names2))
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        #print("dof_names:", self.dof_names)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            # 筛选出「碰撞时需要被惩罚」的部件名称。
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            # 筛选出「碰撞时直接终止当前 episode（回合）」的部件名称
            termination_contact_names.extend([s for s in body_names if name in s])
        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device,
                                                     requires_grad=False)
        claw_names =[]
        for name in self.cfg.asset.claw_names:
            claw_names.extend([s for s in body_names if name in s])
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel

        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        #print("baseinit:", self.base_init_state)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])


        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        #self.rod_handles = []\
        #self.rod_actor_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.rod_handles_all = []
        self.rod_actor_indices_all = torch.full(
            (self.num_envs, self.num_rod_levels), -1, dtype=torch.long, device=self.device
        )
        self.robot_actor_indices = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)

        total_actors=0
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            #print(f"环境{i}的env_handle:", env_handle)#只是告诉 Gym “我不提供包围盒，你按我自己放好的位置来”。你的代码正是这种做法——用 self.env_origins[i] 把每个 env 的东西摆开，所以就算边界是 0，也不会重叠。
            #env 边界不是“墙”，不会阻止跨 env 碰撞。真正避免相互影响，要把不同 env 的 actor 放得足够远（你已经在做）。
            pos = self.env_origins[i].clone()#确定每个环境的原点，但是可能会有一样的，所以下面一行加了变化来保证每个环境原点不一样，但是视觉上可能会有重合，也仅限于地形模式（heightfield和trimesh）
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset,
                                                                i)  # 摩擦随机化每个环境的所有形状同一个摩擦系数，每个环境不同
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i,
                                                 self.cfg.asset.self_collisions, 0)
            #print(f"环境{i}的机器人handle：",actor_handle)
            self.robot_actor_indices[i] = self.gym.get_actor_index(env_handle, actor_handle, gymapi.DOMAIN_SIM)
            #print(f"环境{i}的机器人robot_indices：",self.robot_actor_indices[i])
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            env_rod_handles = []

            dx = (pos[0] - self.env_origins[i, 0]).item()
            dy = (pos[1] - self.env_origins[i, 1]).item()
            dz = (pos[2] - self.env_origins[i, 2]).item()
            x0, y0, z0 = self.cfg.obj.pos

            for lvl, rod_asset in enumerate(rod_assets):
                rod_props = self.gym.get_asset_rigid_shape_properties(rod_asset)
                for s in range(len(rod_props)):
                    rod_props[s].friction = float(self.cfg.obj.friction)
                    rod_props[s].restitution = float(self.cfg.obj.restitution)
                self.gym.set_asset_rigid_shape_properties(rod_asset, rod_props)

                rod_pose = gymapi.Transform()

                # 默认先把所有rod都放远，只激活 level=0 的那根
                if lvl == 0:
                    rod_pose.p = gymapi.Vec3(x0 + dx, y0 + dy, z0 + dz)
                else:
                    rod_pose.p = gymapi.Vec3(1000.0 + 10.0 * lvl, 1000.0 + i, 1000.0)

                rod_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

                if i == 0 and lvl == 0:
                    q = rod_pose.r
                    self.rod_quat = torch.tensor([q.x, q.y, q.z, q.w], device=self.device)

                rod_handle = self.gym.create_actor(
                    env_handle,
                    rod_asset,
                    rod_pose,
                    f"{self.cfg.obj.name}_{lvl}",
                    i,
                    0,
                    0
                )

                env_rod_handles.append(rod_handle)
                self.rod_actor_indices_all[i, lvl] = self.gym.get_actor_index(
                    env_handle, rod_handle, gymapi.DOMAIN_SIM
                )

            self.rod_handles_all.append(env_rod_handles)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )

        self.claw_indices = torch.zeros(
            len(claw_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(claw_names)):
            self.claw_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], claw_names[i]
            )
#             props=rod_shape_props_asset
#             for s in range(len(props)):
#                 props[s].friction = float(self.cfg.obj.friction)
#                 props[s].restitution = float(self.cfg.obj.restitution)
#
#             self.gym.set_asset_rigid_shape_properties(rod_asset, props)
#
#             dx=(pos[0] - self.env_origins[i, 0]).item()
#             dy=(pos[1] - self.env_origins[i, 1]).item()
#             dz=(pos[2] - self.env_origins[i, 2]).item()
#             x0,y0,z0=self.cfg.obj.pos
#             rod_pose=gymapi.Transform()
#             rod_pose.p=gymapi.Vec3(x0+dx,y0+dy,z0+dz)
#             rod_pose.r=gymapi.Quat(0.0,0.0,0.0,1.0)
#             q=rod_pose.r
#             self.rod_quat=torch.tensor([q.x,q.y,q.z,q.w], device=self.device)
#             rod_handle=self.gym.create_actor(env_handle,rod_asset,rod_pose,self.cfg.obj.name,i,0,0)
#             #print(f"环境{i}的handle：",rod_handle)
#             self.rod_handles.append(rod_handle)
#             #print("handels:",self.rod_handles)
#             self.rod_actor_indices[i] = self.gym.get_actor_index(env_handle, rod_handle, gymapi.DOMAIN_SIM)
# #SIM 域（DOMAIN_SIM）索引把所有环境的 actor 按创建顺序接在一起计数：
#             #print(self.rod_actor_indices)
#             dof_props = self._process_dof_props(dof_props_asset, i)
#             #print("关节角限制",dof_props)
#             # 调用方法processdof props调整关节属性增加鲁棒性   读取为了预留可改动空间，给self.doflimit赋值，return的是dofprops
#             self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
#             body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
#             # 获取当前机器人的刚体属性（如质量、惯性、阻尼等）。
#             body_props = self._process_rigid_body_props(body_props, i)
#             # 自定义方法，可能根据环境索引 i 调整刚体属性（如随机化质量分布）。调用方法process rigid body props
#             self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
#             self.envs.append(env_handle)
#             self.actor_handles.append(actor_handle)
#             #env_actors = self.gym.get_actor_count(env_handle)
#             #print(f"环境{i}的actor数: {env_actors}")  # 应输出2
#             #total_actors += env_actors  # 累加
#         # 循环结束后打印全局总数
#         #print(f"全局总actor数: {total_actors}")  # 最终应输出100（50×2）
#         for i in range(len(penalized_contact_names)):
#             self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
#                                                                                       self.actor_handles[0],
#                                                                                       penalized_contact_names[i])
#         #print("penalize:",self.penalised_contact_indices)
#         # 记录所有需要惩罚碰撞的刚体部件的索引（如躯干、手臂等）
#         self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
#                                                        device=self.device, requires_grad=False)
#         for i in range(len(termination_contact_names)):
#             self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
#                                                                                         self.actor_handles[0],
#                                                                                         termination_contact_names[i])
#         #print("termination:",self.termination_contact_indices)
#         self.claw_indices = torch.zeros(len(claw_names), dtype=torch.long, device=self.device, requires_grad=False)
#         for i in range(len(claw_names)):
#             self.claw_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], claw_names[i])
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
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        if self.cfg.obj.curriculum:
            self._update_rod_curriculum(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._apply_rod_level(env_ids)
        self.stand_hold_steps[env_ids] = 0
        self.max_stand_hold_steps[env_ids] = 0
        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        # self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def _update_rod_curriculum(self, env_ids):
        root = self.root_states.view(-1, 13)
        h = root[self.robot_actor_indices[env_ids], 2]

        success = self.max_stand_hold_steps[env_ids] >= self.success_hold_steps
        fail = ~success

        self.rod_levels[env_ids] += success.long()
        self.rod_levels[env_ids] -= fail.long()
        self.rod_levels[env_ids] = torch.clamp(self.rod_levels[env_ids], 0, self.num_rod_levels - 1)

    def _apply_rod_level(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        root = self.root_states.view(-1, 13)

        x0, y0, z0 = self.cfg.obj.pos

        for lvl in range(self.num_rod_levels):
            rod_idx = self.rod_actor_indices_all[env_ids, lvl]

            active = (self.rod_levels[env_ids] == lvl)
            inactive = ~active

            if torch.any(active):
                idx = rod_idx[active]
                eids = env_ids[active]
                root[idx, 0] = x0 + self.env_origins[eids, 0]
                root[idx, 1] = y0 + self.env_origins[eids, 1]
                root[idx, 2] = z0 + self.env_origins[eids, 2]
                root[idx, 3:7] = self.rod_quat
                root[idx, 7:13] = 0.0

            if torch.any(inactive):
                idx = rod_idx[inactive]
                root[idx, 0] = 1000.0 + 10.0 * lvl
                root[idx, 1] = 1000.0
                root[idx, 2] = 1000.0
                root[idx, 3:7] = 0.0
                root[idx, 6] = 1.0
                root[idx, 7:13] = 0.0

        all_rod_idx = self.rod_actor_indices_all[env_ids].reshape(-1).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root),
            gymtorch.unwrap_tensor(all_rod_idx),
            len(all_rod_idx)
        )
    def _reset_root_states(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        root = self.root_states.view(-1, 13)
        #rod_idx=self.rod_actor_indices[env_ids].to(device=self.device, dtype=torch.long)
        robot_idx=self.robot_actor_indices[env_ids].to(device=self.device, dtype=torch.long)

        if self.custom_origins:
            root[robot_idx]=self.base_init_state.to(root.dtype).to(root.device)
            root[robot_idx,:3]+= self.env_origins[env_ids]
            root[robot_idx,:2]+=torch_rand_float(-1.,1.,(len(env_ids), 2), device=self.device)
        else:
            root[robot_idx] = self.base_init_state.to(root.dtype).to(root.device)# self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, actors_per_env, 13)
            root[robot_idx, :3] += self.env_origins[env_ids]
        # 打印关键索引值，检查是否有负

        # 替换原print，先打印形状、设备和是否有NaN
        # print("contact_forces形状:", self.contact_forces.shape)  # 应是(50, N, 3)
        # print("contact_forces设备:", self.contact_forces.device)  # 应与self.device一致（如cuda:0）
        # print("contact_forces是否有NaN:", torch.isnan(self.contact_forces).any().item())  # 应是False
        #
        # print("env_ids:", env_ids)  # 应是0~num_envs-1的整数
        # print("robot_idx:", robot_idx)  # 应是>=0且<total_actors的整数
        # print("rod_idx:", rod_idx)  # 应是>=0且<total_actors的整数
        # print("root总长度:", root.shape[0])  # total_actors = num_envs * 2（因为每个环境2个actor）

        #root[robot_idx,7:13]= torch_rand_float(-1.,1.,(len(env_ids), 6), device=self.device)
        root[robot_idx, 7:13] = torch.zeros_like(root[robot_idx, 7:13])
        #你看到的“随机线速度/角速度”并不是为了让东西乱飞，而是刻意制造轻微扰动，常见目的有这些，现实中不可能每次都完美静止、完美姿态。给初始速度加一点随机扰动，能训练/验证出对小扰动不敏感、能快速收敛/自稳的策略或控制器。
#促进探索（RL 训练）
#完美固定的初始状态会让采样的状态分布很窄，策略容易“过拟合 reset 姿态”。小随机速度能让轨迹覆盖更丰富的状态，提升探索效率，降低局部最优和数据相关性。
        #root[rod_idx,7:13]=torch.zeros_like(root[rod_idx,7:13])#在asset.options里面已经取消了重力
        x0,y0,z0=self.cfg.obj.pos
        #rx,ry=self.cfg.obj.rand_xy
        #dx=torch_rand_float(-rx,rx,(len(env_ids),1), device=self.device)
        #dy=torch_rand_float(-ry,ry,(len(env_ids),1), device=self.device)
        #dx=dx.squeeze(1)
        #dy=dy.squeeze(1)
        # root[rod_idx,0]=x0+self.env_origins[env_ids,0]
        # root[rod_idx,1]=y0+self.env_origins[env_ids,1]
        # root[rod_idx,2]=z0+self.env_origins[env_ids,2]
        # root[rod_idx,3:7]=self.rod_quat
        # root[rod_idx,7:13]=0.0
        robot_idx_i32 = robot_idx.to(dtype=torch.int32)
        #rod_idx_i32 = rod_idx.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(root.to(dtype=torch.float32)),gymtorch.unwrap_tensor(robot_idx_i32), len(robot_idx))
        #self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(root.to(dtype=torch.float32)),gymtorch.unwrap_tensor(rod_idx_i32), len(rod_idx))

    def _claw_contact_flags(self, a=20.0):
        contact_forces_xy = self.contact_forces[..., :2]  # shape: [num_envs, N_contact_points, 2]
        f1= torch.norm(contact_forces_xy, dim=2)  # shape: [num_envs, N_contact_points]

        return (torch.sum(f1[:, self.claw_indices] ,dim=1)>2*a)  ,  (self.contact_forces[:,5,2]>a)

    def _claw_contact_rod(self,threshold=20.0):
        contact1,contact2=self._claw_contact_flags(a=threshold)
        return contact1 & contact2

    def _reward_claw_stand(self):
        root = self.root_states.view(-1, 13)
        h=root[self.robot_actor_indices,2]
        h0=1.3
        k=10
        return h#torch.sigmoid(k*(h-h0))

    def _reward_hold_vel(self):
        # 前两关节速度平方
        v = self.dof_vel[:, :2]
        return self.is_hold* (v * v).sum(dim=1)

    def _reward_h_balance(self):
        root = self.root_states.view(-1, 13)
        h = root[self.robot_actor_indices, 2]  # [num_envs]

        dh = h - self.last_h  # 每步高度变化
        self.last_h = h.detach()  # 只存数值，别连计算图

        # 归一化（你高度范围 0.4~2.2，跨度 1.8）
        dh_norm = dh / 1.8
        jitter = dh_norm * dh_norm  # (Δh)^2

        gate = self.is_hold.float()  # 只在 hold 阶段罚
        return gate * jitter