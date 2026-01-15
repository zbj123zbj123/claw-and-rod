from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Claw4Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 6000
        num_observations = 18
        num_actions = 2

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0., 0.] # min max [m/s]
            lin_vel_y = [-0., 0.]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0,0.4]#[0.0, 0.0, 3.0]#[0.0, 0.0,0.4]#[0.0, 0.0, 2.2]#[0.0, 0.0,0.4]#[0.0, 0.0, 3.0]#[0.0, 0.0,0.4]#[0.0, 0.0, 3.0]# #[0.0, 0.0, 3.0]  # x,y,z [m]
        rot = [0, 1, 0, 0]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
                'P_base_to_1_Link': 0,
            'P_1_to_2_Link': -0.,
            'P_2_to_left_Link': 7.47/180*3.1416,
            'P_2_to_right_Link': 30/180*3.1416,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'P_base_to_1_Link': 600, 'P_1_to_2_Link':500,#刚度大→关节 “执着”→一点偏差就产生大力矩，快速回到目标位置
                     'P_2_to_left_Link': 100., 'P_2_to_right_Link': 100.}  # [N*m/rad]
        damping = {'P_base_to_1_Link': 25, 'P_1_to_2_Link':8.,
                     'P_2_to_left_Link': 2, 'P_2_to_right_Link': 2.}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.2
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_actuator_network = False
        actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/claw4/urdf/claw4.urdf'
        name = "claw4"
        fix_base_link = False
        disable_gravity = False
        collapse_fixed_joints = False
        foot_name = 'None'
        penalize_contacts_on =["P_base_Link","P_1_Link", "P_2_Link","P_left_Link","P_right_Link"]
        terminate_after_contacts_on = ["P_base_Link"]
        claw_names = ["P_left_Link","P_right_Link"]
        default_dof_drive_mode = 3
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
    class obj:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/obj/rod/urdf/rod.urdf'
        name = 'rod'
        pos=[0.0,0.0,1.3]
        rand_xy=[0.05,0.05]
        friction=4.0
        restitution=0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [2,2.2]
        push_robots=True
        max_push_vel_xy = 0.05#0.2

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True#GAI
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = True

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            claw_stand=+20.0#GAI
            h_balance=0
            hold_vel=-0.004
            action_rate = -0.
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            orientation = -0.
            torques = -3e-2
            dof_vel = -0.00000
            dof_acc = -0.0000001
            base_height = -0.
            feet_air_time = 0.
            collision = -0.
            feet_stumble = -0.0
            stand_still = -0.


class Claw4CfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'claww4_grasp_4dof'
        max_iterations = 100000
    class policy:
        init_noise_std = 1.#GAI
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01#GAI