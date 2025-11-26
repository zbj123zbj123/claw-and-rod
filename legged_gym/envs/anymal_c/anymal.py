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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg

class Anymal(LeggedRobot):
    cfg : AnymalCRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.tune_on = True
        self.tune_joint = "P_base_to_1_Link"
        self.tune_A = 1.57
        self.tune_delay = 1.0
        self.tune_f = 0.50
        self.tune_dt = self.sim_params.dt
        self.tune_t = 0.0
        names = self.dof_names
        self.tune_jid = names.index(self.tune_joint)
        self.tune_last_target = self.default_dof_pos.detach().clone()  # 保存上一拍的 pos_target（全关节）
        self.tune_k = 0
        from collections import deque
        self.err_win = deque(maxlen=int(2.0 / self.tune_dt))  # 最近 2 秒的误差窗口
        from torch.utils.tensorboard import SummaryWriter
        import matplotlib
        matplotlib.use("Agg")  # 服务器无显示也能出图
        import matplotlib.pyplot as plt
        self.tb = SummaryWriter("/home/abc/isaacgym/legged_gym/logs/tb_pid_tune")
        self.tr_t, self.tr_ref, self.tr_pos, self.tr_err = [], [], [], []

        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.
        self.prev_rel_height[env_ids] = self._relative_base_height(env_ids).detach()

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)
        self.prev_rel_height = torch.zeros(self.num_envs, device=self.device)

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

    def _relative_base_height(self, env_ids=None):
        h_abs = self.root_states[:, 2]
        ground = None
        if hasattr(self, "_get_heights"):
            g = self._get_heights()
            if torch.is_tensor(g):
                ground = g
        if torch.is_tensor(ground):
            ground = ground.to(device=self.root_states.device, dtype=self.root_states.dtype)
            ground_z = ground.mean(dim=1) if ground.ndim == 2 else ground
        else:
            ground_z = torch.zeros_like(h_abs)

        h_rel = h_abs - ground_z
        return h_rel if env_ids is None else h_rel[env_ids]

    def _feet_contact_flags(self, a=20.0,b=1.0):
        f=torch.norm(self.contact_forces,dim=2)
        lf=self.feet_name_to_idx['LF_FOOT']
        rf=self.feet_name_to_idx['RF_FOOT']
        lh=self.feet_name_to_idx['LH_FOOT']
        rh=self.feet_name_to_idx['RH_FOOT']
        return (f[:,lf]>b), (f[:,rf]>b),(f[:,lh]>a), (f[:,rh]>a)
    def _feet_contact_flags(self, a=20.0, b=1.0):
        f = torch.norm(self.contact_forces, dim=2)
        lf = self.feet_name_to_idx["LF_FOOT"];
        rf = self.feet_name_to_idx["RF_FOOT"]
        lh = self.feet_name_to_idx["LH_FOOT"];
        rh = self.feet_name_to_idx["RH_FOOT"]
        return (f[:, lf] > b), (f[:, rf] > b), (f[:, lh] > a), (f[:, rh] > a)

    def _rear_only(self, threshold1=20.0,threshold2=1.0):
        lf, rf, lh, rh = self._feet_contact_flags(a=threshold1,b=threshold2)
        return (~lf) & (~rf) & lh & rh

    def _rear_only_mask(self):
        # contact_forces: [N, bodies, 3]
        Fz = self.contact_forces[..., 2]
        idx = self.feet_name_to_idx
        rear_F = Fz[:, idx["LH_FOOT"]] + Fz[:, idx["RH_FOOT"]]
        front_F = Fz[:, idx["LF_FOOT"]] + Fz[:, idx["RF_FOOT"]]
        rear_min, front_max, tau = 25.0, 5.0, 15.0
        rear_gate = torch.sigmoid((rear_F - rear_min) / tau)
        front_gate = torch.sigmoid((front_max - front_F) / tau)
        return (rear_gate * front_gate).clamp(0.0, 1.0)

    def _reward_height_if_rear(self):
        h=self._relative_base_height()
        tgt=torch.full_like(h,self.cfg.rewards.base_height_target)
        sig=getattr(self.cfg.rewards,"height_sigma",0.06)
        return torch.exp(-((h-tgt)**2)/(sig**2))*self._rear_only_mask()

    def _reward_height(self, min_h=0.30, scale=1.0):
        h=self._relative_base_height()
        r=torch.relu(h-min_h)*scale
        return r*self._rear_only()

    def _upright_metrics_from_gb(self):
        gb = self.projected_gravity
        gmag = gb.norm(dim=1, keepdim=True).clamp_min(1e-9)
        dot_xu = (gb[:, 0].abs() / gmag[:, 0]).clamp(0., 1.)
        dot_zu = (gb[:, 2].abs() / gmag[:, 0]).clamp(0., 1.)
        return dot_xu, dot_zu

    def _reward_upright_for_rear(self):
        dot_xu, dot_zu = self._upright_metrics_from_gb()
        upright_vertical = (dot_xu ** 2) * (1.0 - dot_zu)
        return upright_vertical * self._rear_only_mask()

    def _reward_termination(self):
        if hasattr(self, "time_out_buf"):
            fail_mask = self.reset_buf & (~self.time_out_buf)
            time_out_mask = self.time_out_buf
        else:
            fail_mask = self.reset_buf
            time_out_mask = None
        f = torch.norm(self.contact_forces, dim=2)  # [N, bodies]
        idx = self.feet_name_to_idx
        fore_force = f[:, idx["LF_FOOT"]] + f[:, idx["RF_FOOT"]]
        rear_force = f[:, idx["LH_FOOT"]] + f[:, idx["RH_FOOT"]]
        h = self._relative_base_height()
        gb = self.projected_gravity
        gmag = gb.norm(dim=1, keepdim=True).clamp_min(1e-9)
        dot_xu = (gb[:, 0].abs() / gmag[:, 0]).clamp(0., 1.)
        dot_zu = (gb[:, 2].abs() / gmag[:, 0]).clamp(0., 1.)
        rear_only = (self._rear_only_mask() > 0.5)
        tilt_bad_general = (dot_zu < 0.5)
        tilt_bad_rear = (dot_xu < 0.766) | (dot_zu > 0.60)
        tilt_bad = torch.where(rear_only, tilt_bad_rear, tilt_bad_general)
        bad_fore = (h > 0.60) & (fore_force > 120.)
        rear_lost = (rear_force < 40.) & (h > 0.45)
        too_low = (h < 0.23)

        extra_fail = bad_fore | rear_lost | too_low | tilt_bad
        fail_mask = fail_mask | extra_fail

        rew = -1.0 * fail_mask.float()
        if time_out_mask is not None:
            rew = rew + 1.0 * time_out_mask.float()
        return rew



    def _reward_rear_weight_bias(self):
        f = torch.norm(self.contact_forces, dim=2)
        idx = self.feet_name_to_idx
        rear = f[:, idx["LH_FOOT"]] + f[:, idx["RH_FOOT"]]
        fore = f[:, idx["LF_FOOT"]] + f[:, idx["RF_FOOT"]]
        bias = (rear - fore) / (rear + fore + 1e-6)
        return torch.clamp(0.5 + 0.5 * bias, 0., 1.)

    def _reward_penalty_fore_foot_force(self, scale=1 / 150.0):
        f = torch.norm(self.contact_forces, dim=2)
        idx = self.feet_name_to_idx
        fore = f[:, idx["LF_FOOT"]] + f[:, idx["RF_FOOT"]]
        return - torch.clamp(fore * scale, 0., 1.5)


    def _reward_com_over_rear_support(self):
        base_xy = self.root_states[:, :2]
        lh = self.feet_name_to_idx["LH_FOOT"];
        rh = self.feet_name_to_idx["RH_FOOT"]
        feet_xy = self.rigid_body_states[:, :, :2]
        p_lh = feet_xy[:, lh, :];
        p_rh = feet_xy[:, rh, :]
        v = p_rh - p_lh
        v2 = (v * v).sum(dim=1) + 1e-6
        t = ((base_xy - p_lh) * v).sum(dim=1) / v2
        t = torch.clamp(t, 0., 1.)
        proj = p_lh + t.unsqueeze(1) * v
        dist = (base_xy - proj).norm(dim=1)

        sig = 0.07
        bonus = torch.exp(-(dist ** 2) / (sig ** 2))

        return bonus * self._rear_only_mask()

    def _reward_forbidden_contacts(self, force_thresh=30.0, scale=1 / 300.0, max_penalty=2.0):
        F = torch.norm(self.contact_forces, dim=2)
        F_forbid = F.index_select(1, self.forbidden_contact_ids)
        excess = torch.relu(F_forbid - force_thresh)
        raw = excess.sum(dim=1) * scale
        pen = -torch.clamp(raw, min=0.0, max=max_penalty)
        return pen







