from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# LP = {"Environment": {"map_dimensions": [{"length": 22, "width": 22}], "obstacles": [{"name": "obs_1", "position": [4, 4], "size": [3, 3, 99]}, {"name": "obs_2", "position": [9, 9], "size": [2, 2, 8]}, {"name": "obs_3", "position": [14, 3], "size": [3, 2, 7]}, {"name": "obs_4", "position": [7, 13], "size": [4, 3, 99]}, {"name": "obs_5", "position": [18, 18], "size": [2, 2, 4]}, {"name": "obs_6", "position": [10, 1], "size": [1, 1, 99]}, {"name": "obs_7", "position": [3, 18], "size": [3, 3, 99]}, {"name": "obs_8", "position": [1, 8], "size": [2, 4, 10]}, {"name": "obs_9", "position": [13, 1], "size": [2, 2, 99]}, {"name": "obs_10", "position": [20, 15], "size": [2, 2, 5]}], "robot_positions": [{"start_position": [1, 1], "end_position": [20, 20]}]}}

class H1RoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        #num_envs = 256
        num_envs=4
        num_observations = 163
        num_actions = 10
     
    class terrain( LeggedRobotCfg.terrain):
        water_terrain_random_scale = [0.8, 1.2]
        apply_water = True
        water_attitude = 0.3
        mesh_type = 'trimesh'
        measured_points_x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5] # 1mx1m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]

        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        terrain_proportions = [0.1, 0.1, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1]
        

    class init_state( LeggedRobotCfg.init_state ):
        #x = -LP["Environment"]["map_dimensions"][0]["width"] / 2 + LP["Environment"]["robot_positions"][0]["start_position"][0]
        #y = -LP["Environment"]["map_dimensions"][0]["length"] / 2 + LP["Environment"]["robot_positions"][0]["start_position"][1]
        x = 0
        y = 0
        pos = [x, y, 1.0] # x,y,z [m]
        default_joint_angles = {
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.4,         
           'left_knee_joint' : 0.8,       
           'left_ankle_joint' : -0.4,

           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.4,                                       
           'right_knee_joint' : 0.8,                                             
           'right_ankle_joint' : -0.4,

           'torso_joint' : 0.,
           
           'left_shoulder_pitch_joint' : 0., 
           'left_shoulder_roll_joint' : 0, 
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint'  : 0.,

           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : 0.0,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 0.,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            'hip_yaw': 200,
            'hip_roll': 200,
            'hip_pitch': 200,
            'knee': 300,
            'ankle': 40,
            'torso': 300,
            'shoulder': 100,
            "elbow":100,
            }  # [N*m/rad]
        damping = {
            'hip_yaw': 5,
            'hip_roll': 5,
            'hip_pitch': 5,
            'knee': 6,
            'ankle': 2,
            'torso': 6,
            'shoulder': 2,
            "elbow":2,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.15
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1.urdf'
        name = "H1"
        foot_name = "ankle"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.98
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -3.5e-8
            feet_air_time = 1.0
            collision = 0.0
            action_rate = -0.01
            torques = 0.0
            dof_pos_limits = -10.0
            #stand_still = -0.1
            #dis_to_goal = -0.1

class H1RoughCfgPPO( LeggedRobotCfgPPO ):
    
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_H1'
        num_steps_per_env = 96 # per iteration
        max_iterations = 5000 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations

    class algorithm( LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01


  
