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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class CassieRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 128
        num_observations = 253
        num_actions = 12

    
    class terrain( LeggedRobotCfg.terrain):
        water_terrain_random_scale = [0.8, 1.2]
        apply_water = False
        water_attitude = 0.3
        mesh_type = 'trimesh'     # none, plane, heightfield or trimesh
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None
        # terrain_kwargs = {'type': 'random_uniform',
        #                   'terrain_kwargs': {
        #                       'min_height': -0.2, 
        #                       'max_height': 0.2, 
        #                       'step': 0.2, 
        #                       'downsampled_scale': 0.5
        #                       }
        #                     }
        # terrain_kwargs = {'type': 'sloped',
        #                   'terrain_kwargs': {
        #                       'slope': -0.5
        #                       }
        #                     }
        # terrain_kwargs = {'type': 'pyramid_sloped',
        #                   'terrain_kwargs': {
        #                       'slope': -0.5
        #                       }
        #                     }
        # terrain_kwargs = {'type': 'discrete_obstacles', 
        #                   'terrain_kwargs': {
        #                       'max_height': 0.5, 
        #                       'min_size': 1., 
        #                       'max_size': 5., 
        #                       'num_rects': 20
        #                       }
        #                     }
        # terrain_kwargs = {'type': 'wave',
        #                   'terrain_kwargs': {
        #                       'num_waves': 2., 
        #                       'amplitude': 1.
        #                       }
        #                     }
        # terrain_kwargs = {'type': 'stairs',
        #                   'terrain_kwargs': {
        #                       'step_width': 0.75, 
        #                       'step_height': -0.5
        #                       }
        #                     }
        # terrain_kwargs = {'type': 'pyramid_stairs',
        #                   'terrain_kwargs': {
        #                       'step_width': 0.75, 
        #                       'step_height': -0.5
        #                       }
        #                     }
        # terrain_kwargs = {'type': 'stepping_stones',
        #                   'terrain_kwargs': {
        #                       'stone_size': 0.3, 
        #                       'stone_distance': 0.3, 
        #                       'max_height': 0.5, 
        #                       'platform_size': 0.
        #                       }
        #                     }
        # terrain_kwargs = {'type':'pillars',
        #                   'terrain_kwargs': {
        #                       'num_pillars': 5, 
        #                       'max_pillar_size': 1.5,
        #                       'pillar_gap': 3.0,
        #                       'step_height': 0.35
        #                       }
        #                     }
        # terrain_kwargs = {'type':'gap',
        #                   'terrain_kwargs': {
        #                       'gap_size': 0.7, 
        #                       'platform_size': 3.
        #                       }
        #                     }
        # terrain_kwargs = {'type':'pit',
        #                   'terrain_kwargs': {
        #                       'depth': 0.7, 
        #                       'platform_size': 4.
        #                       }
        #                     }

        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 20 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        terrain_proportions = [0.1, 0.1, 0.3, 0.2, 0.15, 0.05, 0.05, 0.05]

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,

            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {   'hip_abduction': 100.0, 'hip_rotation': 100.0,
                        'hip_flexion': 200., 'thigh_joint': 200., 'ankle_joint': 200.,
                        'toe_joint': 40.}  # [N*m/rad]
        damping = { 'hip_abduction': 3.0, 'hip_rotation': 3.0,
                    'hip_flexion': 6., 'thigh_joint': 6., 'ankle_joint': 6.,
                    'toe_joint': 1.}  # [N*m*s/rad]     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf'
        name = "cassie"
        foot_name = 'toe'
        shank_name = "tarsus" # 小腿
        thigh_name = "thigh" # 大腿
        penalize_contacts_on = ["SHANK", "THIGH"]
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            collision = -0.
            tracking_lin_vel = 10.0
            tracking_ang_vel = 1.0
            torques = -5.e-6
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.

class CassieRoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_cassie'

        num_steps_per_env = 96 # per iteration
        max_iterations = 3000 # number of policy updates
        # logging
        save_interval = 300 # check for potential saves every this many iterations

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

# class CassieRoughCfgDM(LeggedRobotCfgDM):
    
#     class runner(LeggedRobotCfgDM.runner):
#         run_name = ''
#         experiment_name = 'rough_cassie'

#         num_steps_per_env = 96 # per iteration
#         max_iterations = 10000 # number of policy updates
#         # logging
#         save_interval = 1000 # check for potential saves every this many iterations

#     class algorithm(LeggedRobotCfgDM.algorithm):
#         entropy_coef = 0.01



  