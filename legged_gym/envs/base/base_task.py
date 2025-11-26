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

import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
#关闭 PyTorch JIT 的 profiling 相关逻辑（trace 时的热点探测与基于 profile 的执行器）。
#为什么：在高频率、短步长的仿真-学习循环里（每步很短），JIT 的 profiling 反而可能带来额外的开销与不稳定抖动；关掉更稳。
        # allocate buffers

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        #obs_buf：形状 [num_envs, num_obs]，浮点。保存每个并行环境的观测向量。
#rew_buf：形状 [num_envs]，浮点。保存每个环境当前步的即时回报。
#reset_buf：形状 [num_envs]，整型（long）。1 表示该 env 需要重置。
#这里在 __init__ 里不一定初始化，但后面常见做法是初始置 1，保证开局 reset() 会重置全部环境（你的片段中是在别处初始化为 1，若没有，reset() 也会主动重置全部）。

#episode_length_buf：形状 [num_envs]，long。计数该回合已走的步数（用于到时终止等）。
#time_out_buf：形状 [num_envs]，bool。True 表示“到达最大步数”的超时（与摔倒/越界等失败终止区分，方便算法不同处理）。

#device：都放在 self.device 上（可能是 cuda:0 或 cpu），取决于你之前的 GPU pipeline 选择。放 GPU可减少 CPU↔GPU 拷贝。

#dtype 的讲究：

#float：连续量（观测、奖励）。

#long：标志/计数常用 long，且便于当做索引。

#bool：语义更清晰（比如 time_out_buf）。
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None
            # self.num_privileged_obs = self.num_obs把 num_privileged_obs 退化为 num_obs”是另一种做法
#特权观测（asymmetric actor-critic）：
#若配置里给了 num_privileged_obs，则为critic 专用的更“上帝视角”的观测分配缓冲（训练更稳）。
#否则置 None，上层 runner 会据此判断是否走不对称训练。
        self.extras = {}
#“给记录器/可视化看的临时信息”为什么不直接改 rew_buf/reset_buf

#rew_buf、reset_buf 是算法接口（训练必须用的），不能随便塞别的。

#extras 是纯附加、不影响梯度的；你想看什么就放什么，方便调参与排错。
        # create envs, sim and viewer
        self.create_sim()#create_sim()：抽象方法由子类实现，通常里头会：

#self.sim = gym.create_sim(...) 创建仿真；

#创建地形、并行 env、放置 actor（机器人）、获取各种句柄；

#建立状态/动作的张量映射。
        self.gym.prepare_sim(self.sim)#通知底层完成一些资源准备与缓存构建，在大规模并行时能减少后续卡顿

        # todo: read from config
        self.enable_viewer_sync = True
#enable_viewer_sync：是否按真实时间节流渲染（True → 每帧 sync_frame_time，能看到“正常速度”的动画；False → 不限速，适合快速刷步骤
        self.viewer = None
#viewer：无头模式（headless）下保持 None。
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
#create_viewer：创建一个窗口（OpenGL/DirectX 上下文），可交互看仿真。
#键盘绑定：
#ESC → 发布 "QUIT" 事件（退出）。
#V → 发布 "toggle_viewer_sync" 事件（切换是否限速同步）。
    def get_observations(self):
        return self.obs_buf
    #观测读取接口
    def get_privileged_observations(self):
        return self.privileged_obs_buf
#做了什么：对外暴露当前观测（以及可能的特权观测）。
#为什么：上层 runner 有时需要直接拿 buffer（比如复位后第一次读）。
    def reset_idx(self, env_ids):
       #复位接口（抽象与通用全量复位） """Reset selected robots"""
        raise NotImplementedError
#你手动调用 env.reset() 时（它会把 全部 env 的索引 [0..num_envs-1] 传给 reset_idx）。
#训练中每步 step() 后，算法根据 reset_buf==1 的那些环境只重置这些 env，也会调用 reset_idx(done_env_ids)。
    #raise 是什么语法？
    #raise 是抛出异常的语句。一旦执行到raise，当前函数会立即中断，异常沿着调用栈向上冒泡，直到被try/ except 捕获；若没人捕获，程序就报错退出。
#NotImplementedError 用在什么时候？多用于基类或接口中的“占位方法”：告诉你：“这个方法必须由子类重写，否则就报错。”
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
#构造 env 索引：torch.arange(self.num_envs) 生成 [0, 1, ..., num_envs-1]，表示“全部环境”。
#放到同一设备：device=self.device（可能是 'cuda:0' 或 'cpu'），避免后续子类里用到这些索引时发生不必要的设备拷贝/类型转换。
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
#为什么复位后还要走一步？
#为了让“复位后的观测”与“正常 roll-out 的观测”走同一条代码路径（都从 step() 产出）。许多实现把观测更新放在 post_physics_step() 或 step() 内部，如果不走这一步，你拿不到一套与训练时一致的观测。
#下划线 _：表示忽略返回的后 3 个量（即时奖励、done 标志、诊断信息），复位阶段暂时不需要它们。
    def step(self, actions):
        #步进接口
        raise NotImplementedError
#子类实现动作→仿真→读状态→算奖励/终止的全流程，按约定返回：return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
#做了什么：如果窗口被用户点 X 关了，直接退出进程（训练/演示常见做法，粗暴但简单）。
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
#self.gym.query_viewer_action_events(self.viewer)含义：从 viewer 取出自上次查询以来的所有输入事件（键盘/鼠标被你订阅过的那些）。
#与上文对应
#返回：一个可迭代的事件列表，每个 evt 至少有两个常用字段：
#evt.action：事件名（就是你在订阅时传的那个字符串，比如 "QUIT"、"toggle_viewer_sync"）。
#evt.value：事件值。对键盘来说，通常 按下时 > 0（一般为 1），抬起时 == 0。所以这段代码用 > 0 来判定“这是一次按下”。
#事件分发：
#收到 "QUIT" → 退出；
#收到 "toggle_viewer_sync" → 切换限速开关（与 V 键绑定）。
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
#为什么：GPU 仿真（异步）时，需要 fetch_results(sim, True) 阻塞等待上一个 simulate() 完成，确保当前帧绘制的是完整的最新状态。
#注意：如果仿真跑在 CPU，通常不需要这一步（因此判断 self.device != 'cpu'）。
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)
#同步模式（限速）：step_graphics：推进图形流水线；draw_viewer：把当前帧画出来；sync_frame_time（当 sync_frame_time=True）按真实时间节流，让动画“看起来像真实速度”。
#非同步模式（不限速）：仅 poll_viewer_events，不画，不限速（或由其他地方控制绘制），适合追求速度的场景。
#render() 只负责取结果+画，真正的 simulate() 通常在 step() 里做。如果你在 render() 前没有做过 simulate()，那这一帧不会有新的物理结果可画。