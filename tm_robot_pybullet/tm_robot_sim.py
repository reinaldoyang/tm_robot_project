import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import math
import random
import cv2 as cv
import json

def random_spawn_pos(x_range, y_range, z):
    """
    Generate a random position within range
    x_range need to be array-like: [min, max]
    """
    x = random.uniform(x_range[0], x_range[1])
    if x < 0.7:
        # Shrink y_range to 80%
        y_center = (y_range[0] + y_range[1]) / 2
        y_span = (y_range[1] - y_range[0]) * 0.8 / 2  # half of 80% span
        y_min = y_center - y_span
        y_max = y_center + y_span
    else:
        y_min, y_max = y_range[0], y_range[1]
    
    y = random.uniform(y_min, y_max)
    return [x, y, z]

def draw_boundary(x_range, y_range, z):
    """
    Draw a boundary box to see spawn area
    """
    corners = [
        [x_range[0], y_range[0], z], #bottom left
        [x_range[1], y_range[0], z], #bottom right
        [x_range[1], y_range[1], z], #top right
        [x_range[0], y_range[1], z] #top left
    ]
    for i in range(len(corners)):
        p.addUserDebugLine(corners[i], corners[(i+1) % len(corners)], [1, 0, 0], lineWidth = 2) #1,0,0 is color

def create_simulation_env(mode):
    """
    Create the simulation environment with TM robot, table, cube, tray, and gripper
    """
    if mode == "GUI":
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) #get the default pybullet assets
    p.setGravity(0, 0, -10)
    plane_id = p.loadURDF("plane.urdf")
    robot_id = p.loadURDF(
        "tm5-700-nominal.urdf",
        basePosition=[1.4, -0.2, 0.6],
        baseOrientation=[0, 0, 0, 1],
        useFixedBase = True
    )
    table_id = p.loadURDF(
        "table/table.urdf",
        basePosition=[1.0, -0.2, 0],
        baseOrientation = [0, 0, 0.7071, 0.7071]
    )
    # define the spawn range on the table
    x_range = [0.65, 1.2]
    y_range = [-0.7, 0.3]
    z = 0.65  # cube height on top of the table
    draw_boundary(x_range, y_range, z)
    cube_pos = random_spawn_pos(x_range, y_range, z)
    tray_pos = random_spawn_pos(x_range, y_range, z)
    min_distance = 0.15
    while np.linalg.norm(np.array(cube_pos[:2]) - np.array(tray_pos[:2])) < min_distance:
        cube_pos = random_spawn_pos(x_range, y_range, z)

    base_pos, _ = p.getBasePositionAndOrientation(robot_id)
    vec = np.array(cube_pos) - np.array(base_pos)
    distance_xy = np.linalg.norm(vec[:2])
    print(f"Distance from robot base to cube: {distance_xy}")
    cube_id = p.loadURDF(
        "cube.urdf",
        basePosition=cube_pos,
        globalScaling=0.05
    )
    tray_id = p.loadURDF(
        "tray/tray.urdf", 
        tray_pos,
        globalScaling=0.3
    )
    gripper_id = p.loadSDF("gripper/wsg50_one_motor_gripper_new_free_base.sdf")[0]
    return plane_id, robot_id, table_id, cube_id, tray_id, gripper_id

def attach_gripper_to_robot(robot_id, gripper_id):
    """
    Attach the default wsg50 gripper to tm robot
    """
    p.createConstraint(robot_id, 6, gripper_id, 0, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0]) #attach gripper to tmrobot
    tmrobot_cid2 = p.createConstraint(gripper_id, 4, gripper_id, 6, jointType=p.JOINT_GEAR, jointAxis=[1,1,1], parentFramePosition=[0,0,0], childFramePosition=[0,0,0])
    p.changeConstraint(tmrobot_cid2, gearRatio=-1, erp=0.5, relativePositionTarget=0, maxForce=200)

def move_robot(robot_id, gripper_id, arm_joints, end_effector_idx, target_pos, gripper_val):
    """
    move robot to target position with gripper open/close
    """
    # print(f"x value: {target_pos[0]}")
    if 0.63 < target_pos[0] <= 0.65:
        roll = -math.pi/11
        pitch = 1.01*math.pi
        yaw = math.pi/2
        target_orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    elif 0.60 <= target_pos[0] <= 0.63:
        roll = -math.pi/20
        pitch = 1.01*math.pi
        yaw = math.pi/2
        target_orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    elif 0.65 <= target_pos[0] <= 0.7:
        roll = -math.pi/15
        pitch = 1.01*math.pi
        yaw = math.pi/2
        target_orn = p.getQuaternionFromEuler([roll, pitch, yaw])
    else:
        target_orn = p.getQuaternionFromEuler([0, 1.01*math.pi, math.pi/2])

    joint_poses = p.calculateInverseKinematics(robot_id, end_effector_idx, target_pos, target_orn) #ik return 6 for joint states

    for i, j in enumerate(arm_joints):
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=j,
            controlMode=p.POSITION_CONTROL,
            targetPosition=joint_poses[i],
            force=200,
            maxVelocity=3
    )
    #for controlling gripper to close or open
    p.setJointMotorControl2(gripper_id, 4, p.POSITION_CONTROL, targetPosition=gripper_val*0.05, force=100)
    p.setJointMotorControl2(gripper_id, 6, p.POSITION_CONTROL, targetPosition=gripper_val*0.05, force=100)
    
def capture_image(cam_width, cam_height):
    """
    Capture image from a predefined camera view and write to video
    """
    cam_target_pos = [1.2, -0.2, 0.8]
    cam_distance = 1.4
    cam_yaw, cam_pitch, cam_roll = -90, -20, 0
    cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = 2, 0.01, 100, 60
    cam_view_matrix = p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll, cam_up_axis_idx)
    cam_projection_matrix = p.computeProjectionMatrixFOV(cam_fov, cam_width*1./cam_height, cam_near_plane, cam_far_plane)
    image = p.getCameraImage(cam_width, cam_height, cam_view_matrix, cam_projection_matrix)[2][:, :, :3]
    return np.ascontiguousarray(image)

def check_robot_joint_info(robot_id):
    """
    Print out joint info for debug
    """
    num_joints = p.getNumJoints(robot_id)
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        print(j, info[1].decode("utf-8"), info[2])

def check_task_success(cube_pos, tray_pos, max_xy_dist=0.35):
    """
    Takes in cube and tray pos, and max distance betweeen cube and tray center for eval
    """
    xy_dist = np.linalg.norm(np.array(cube_pos[:2]) - np.array(tray_pos[:2]))
    # Check if cube is below tray rim and close enough in xy
    print(f"Distance between cube and tray: {xy_dist}")
    print(f"Height of the cube now: {cube_pos[2]}")
    return xy_dist <= max_xy_dist and cube_pos[2] > 0.66

def get_end_effector_state(robot_id, end_effector_idx):
    """
    Get the end effector pos and orientation
    """
    ee_state = p.getLinkState(robot_id, end_effector_idx, computeForwardKinematics=True)
    ee_pos = ee_state[0]
    ee_orn = ee_state[1]
    return ee_pos, ee_orn

def get_object_state(object_id):
    """
    Get the object pos and orientation
    """
    obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)
    return obj_pos, obj_orn

def get_gripper_tcp(gripper_id, left_tip_index, right_tip_index):
    """
    Compute the gripper TCP posiiton and orientation
    """
    left_tip_pos = p.getLinkState(gripper_id, left_tip_index)[0]
    right_tip_pos = p.getLinkState(gripper_id, right_tip_index)[0]
    tcp_pos = [(l+r)/2 for l, r in zip(left_tip_pos, right_tip_pos)]
    return tcp_pos

def check_grasp_success(tcp_pos, cube_pos, max_dist = 0.2):
    """
    Check if the cube is successfully grasped by the gripper
    """
    dist_to_ee = np.linalg.norm(np.array(cube_pos) - np.array(tcp_pos))
    print(f'Distance to end effector: {dist_to_ee}')
    return dist_to_ee <= max_dist

def save_rlds_episode(base_dir, episode_id, frames, ee_states):
    """
    Save image observation state and also 7D robot state ( EE pose + gripper for a given timestep)
    """
    episode_dir = os.path.join(base_dir, episode_id)
    if not os.path.exists(episode_dir):
        os.makedirs(episode_dir)

    #save image in another folder
    img_dir = os.path.join(episode_dir, "img")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_filenames = []
    for t, img in enumerate(frames):
        img_filename = f"img_{t:04d}.png"
        cv.imwrite(os.path.join(img_dir, img_filename), img)
        img_filenames.append(img_filename)
    
    #save in RLDS format
    episode_data = {
        "episode_id" : episode_id, 
        "timesteps": len(frames),
        "img_filenames": img_filenames,
        "ee_states": ee_states #7D : [x,y,z,rx,ry,rz,gripper_state]
    }
    with open(os.path.join(episode_dir, f"{episode_id}.json"), "w") as f: #open the episode.json file and write the episode data
        json.dump(episode_data, f, indent=2)

def move_and_get_gripper(robot_id, gripper_id, arm_joints, ee_idx, target_pos, gripper_val):
    """
    Extract gripper_val
    """
    move_robot(robot_id, gripper_id, arm_joints, ee_idx, target_pos, gripper_val)
    return gripper_val

def run_simulation():
    """
    Run the simulation with the predefined environments
    """
    plane_id, robot_id, table_id, cube_id, tray_id, gripper_id = create_simulation_env("others")
    attach_gripper_to_robot(robot_id, gripper_id)

    cam_width, cam_height = 224, 224
    end_effector_idx = 6
    arm_joints = [1, 2, 3, 4, 5, 6] #tmrobot movable joints
    
    frames = []
    ee_states = []
    
    cube_pos, cube_orn = get_object_state(cube_id)
    tray_pos, tray_orn = get_object_state(tray_id)

    for t in range(950):
        print(f'\rtimestep {t}...', end='')
        p.stepSimulation()
        time.sleep(1/240) #control simulation loop, 240hz default physics engine

        gripper_val=0
        if t >= 100 and t < 200:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [cube_pos[0], cube_pos[1], 1.1], 0) #move above cube
        elif t >= 200 and t < 300:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [cube_pos[0], cube_pos[1], 0.9], 0) #move down to cube
        elif t >= 300 and t < 400:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [cube_pos[0], cube_pos[1], 0.9], 1) #close gripper
        elif t >= 400 and t < 500:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [cube_pos[0], cube_pos[1], 1.1], 1) #lift up
        elif t >=500 and t < 600:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [tray_pos[0], tray_pos[1], 1.1], 1) #move above tray
        elif t >= 600 and t < 700:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [tray_pos[0], tray_pos[1], 0.95], 1) #move down to tray
        elif t >= 700 and t < 800:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [tray_pos[0], tray_pos[1], 0.95], 0) #open gripper
        elif t >= 800 and t < 900:
            gripper_val = move_and_get_gripper(robot_id, gripper_id, arm_joints, end_effector_idx, [tray_pos[0], tray_pos[1], 1.1], 0) # go up
        
        if t % 48 == 0:
            img = capture_image(cam_width, cam_height)
            frames.append(img)
            #get ee position + orientation + gripper state
            ee_pos, ee_orn = get_end_effector_state(robot_id, end_effector_idx)
            ee_rpy = p.getEulerFromQuaternion(ee_orn)
            ee_states.append([*ee_pos, *ee_rpy, gripper_val]) # * unpacking syntax, not pointer

        if t == 700:
            cube_now, _ = get_object_state(cube_id)
            tcp_pos = get_gripper_tcp(gripper_id, 5, 7)
            grasp_success = check_grasp_success(tcp_pos, cube_now)
            print(f'Grasp success/fail: {grasp_success}')

    cube_now, _ = get_object_state(cube_id)
    task_success = check_task_success(cube_now, tray_pos)
    print(f'Task success/no: {task_success}')
    
    if grasp_success and task_success:
        print("Episode will be saved")
    else:
        print("Episode will NOT be saved")

    p.disconnect()
    return grasp_success, task_success, frames, ee_states

if __name__ == "__main__":
    run_simulation()