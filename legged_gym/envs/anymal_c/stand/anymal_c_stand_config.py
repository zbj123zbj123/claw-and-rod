from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AnymalCStandUpCfg( AnymalCRoughCfg):
    class commands(AnymalCRoughCfg.commands):
        heading_command = False
        resampling_time = 10.0
        class ranges(AnymalCRoughCfg.commands.ranges):
            lin_vel_x =(0.0,0.0)
            lin_vel_y =(0.0,0.0)
            ang_vel_yaw =(0.0,0.0)

    class init_state(AnymalCRoughCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        default_joint_angles = {
            "LF_HAA": 0.0, "LH_HAA": 0.15, "RF_HAA": -0.0, "RH_HAA": -0.15,
            "LF_HFE": 0.35, "LH_HFE": -0.35,
            "RF_HFE": 0.35, "RH_HFE": -0.35,
            "LF_KFE": -0.70, "LH_KFE": 0.70,
            "RF_KFE": -0.70, "RH_KFE": 0.70,
        }
    class terrain(LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = True
        static_friction = 1.2

    class control(AnymalCRoughCfg.control):
        action_scale = 0.45
        stiffness = {'HAA': 80., 'HFE': 110., 'KFE': 110.}
        damping   = {'HAA': 4.0, 'HFE': 6.0, 'KFE': 6.0}
    class rewards(AnymalCRoughCfg.rewards):
        base_height_target = 0.70
        height_sigma = 0.12
        only_positive_rewards = False
        max_contact_force = 500.

        class scales(AnymalCRoughCfg.rewards.scales):
            termination = -5.0
            rear_weight_bias = +5.0
            height_if_rear = +12.0
            upright_for_rear = +20.0
            com_over_rear_support = +25.0
            penalty_fore_foot_force=+5.0
            forbidden_contacts=+5.0

            orientation = 0.0
            base_height = -1.0
            stand_still = -0.0

            torques = -5e-6
            dof_vel = -5e-4
            dof_acc = -1e-7
            action_rate = -0.0005
            ang_vel_xy = 0.0
            lin_vel_z = 0.0
            collision=-0.2

            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            feet_stumble = 0.0
    class domain_rand( AnymalCRoughCfg.domain_rand):
        randomize_base_mass = False

class AnymalCStandUpCfgPPO(AnymalCRoughCfgPPO):
        class runner(AnymalCRoughCfgPPO.runner):
            run_name = 'anymal_c_stand'
            experiment_name = 'anymal_c_stand'
            load_run = -1