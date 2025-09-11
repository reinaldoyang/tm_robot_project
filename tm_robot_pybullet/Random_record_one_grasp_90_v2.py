# latest version: 1000筆資料由此code生成
# success rate = 34.04%
import pybullet as p
import pybullet_data
import numpy as np
import os
import json
import time
import cv2
import random

# === 1. 初始化模擬器與場景 ===
def random_position_around_arm(radius_range=(0.55, 0.75), z=0.025):
    r = random.uniform(*radius_range)
    theta = random.uniform(1.5 * np.pi, 2.5 * np.pi)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return [x, y, z]

def setup_pybullet(gui=True):
    mode = p.GUI if gui else p.DIRECT
    p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # p.loadURDF("plane.urdf")
    # p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])

    p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.62], baseOrientation=p.getQuaternionFromEuler([0, 0, np.pi / 2]), globalScaling=1.0)
    
    robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

    # 使用圓形隨機範圍
    cube_start_pos = random_position_around_arm(z=0.025)
    box_start_pos = random_position_around_arm(z=0.02)
    
    while np.linalg.norm(np.array(box_start_pos[:2]) - np.array(cube_start_pos[:2])) < 0.20:
        box_start_pos = random_position_around_arm()

    cube_id = p.loadURDF("cube_small.urdf", basePosition=cube_start_pos)
    box_id = p.loadURDF("tray/traybox.urdf", basePosition=box_start_pos, globalScaling=0.3)

    return robot_id, cube_id, box_id, cube_start_pos, box_start_pos


# === 2. 擷取影像 ===
def capture_image(width=256, height=256):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.6, 0, 0],
        distance=1.2,
        yaw=90,
        pitch=-90,
        roll=0,
        upAxisIndex=2)
    proj_matrix = p.computeProjectionMatrixFOV(fov=65, aspect=1.0, nearVal=0.1, farVal=3.1)
    _, _, px, _, _ = p.getCameraImage(width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    rgb_array = np.reshape(px, (height, width, 4))[:, :, :3]
    return rgb_array

# === 3. 控制 Panda 前往特定目標 ===
def move_panda(robot_id, target_pos, gripper_closed):
    joint_poses = p.calculateInverseKinematics(robot_id, 11, targetPosition=target_pos)
    for i in range(7):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, joint_poses[i], force=450)  # 手臂的最大旋轉扭力
    # 抓子開關大小 [0.0, 0.8]: [全關, 全開]
    grip_pos = 0.04 if not gripper_closed else 0.0
    for j in [9, 10]:
        p.setJointMotorControl2(robot_id, j, p.POSITION_CONTROL, grip_pos, force=50)    # 夾的力

# === 4. 判斷抓取與任務是否成功 ===
def is_grasp_success(cube_id, ee_pos, threshold=0.05):
    cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
    height_success = cube_pos[2] > 0.05
    dist_to_ee = np.linalg.norm(np.array(cube_pos) - np.array(ee_pos))
    return height_success and dist_to_ee < 0.1

def is_task_success(cube_pos, tray_pos, tray_radius=0.2):
    xy_dist = np.linalg.norm(np.array(cube_pos[:2]) - np.array(tray_pos[:2]))
    return cube_pos[2] < 0.2 and xy_dist < tray_radius

# === 5. 儲存資料 ===
def save_trajectory(save_path, metadata, images):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "meta.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(save_path, f"rgb_{i:03d}.png"), img)

# === 6. 主程式：錄製一筆資料 ===
def record_one(save_path="random_grasp_and_place_dataset/trajectory_0001", gui=True):
    robot, cube, box, cube_pos, box_pos = setup_pybullet(gui=gui)
    traj = {"trajectory_id": "traj_0001", "task": "Pick white cube and place into black box.", "frames": []}
    images = []
    grasp_success = False
    task_success = False

    approach_z = cube_pos[2] + 0.12  # 從上方接近
    close_z = cube_pos[2] + 0.01     # 快貼住
    touch_z = cube_pos[2]            # 幾乎接觸

    for step in range(100):  # 延長模擬時間
        ### grasp邏輯 ###
        if step < 20:
            move_panda(robot, [cube_pos[0], cube_pos[1], 0.10], False)
        elif step < 35:
            move_panda(robot, [cube_pos[0], cube_pos[1], 0.03], False)
        elif step < 45:
            move_panda(robot, [cube_pos[0], cube_pos[1], 0.01], True)  # 關爪
        ### place邏輯 ###
        elif step < 70:
            move_panda(robot, [box_pos[0], box_pos[1], 0.2], True)   # 移至 box 上方
        elif step < 80:
            move_panda(robot, [box_pos[0], box_pos[1], 0.03], True)   # 慢慢下降但不放
        elif step < 90:
            move_panda(robot, [box_pos[0], box_pos[1], 0.03], False)  # 放開持續幾個 step
        else:
            move_panda(robot, [box_pos[0], box_pos[1], 0.15], False)  # 放開後上升


    # for step in range(100):
    #     if step < 20:
    #         move_panda(robot, [cube_pos[0], cube_pos[1], cube_pos[2] + 0.20], False)
    #     elif step < 30:
    #         move_panda(robot, [cube_pos[0], cube_pos[1], cube_pos[2] + 0.05], False)
    #     elif step < 40:
    #         move_panda(robot, [cube_pos[0], cube_pos[1], cube_pos[2] + 0.01], True)
    #     elif step < 60:
    #         move_panda(robot, [box_pos[0], box_pos[1], box_pos[2] + 0.20], True)
    #     elif step < 75:
    #         move_panda(robot, [box_pos[0], box_pos[1], box_pos[2] + 0.01], True)
    #     elif step < 85:
    #         move_panda(robot, [box_pos[0], box_pos[1], box_pos[2] + 0.01], False)
    #     else:
    #         move_panda(robot, [box_pos[0], box_pos[1], box_pos[2] + 0.20], False)

        p.stepSimulation()
        time.sleep(0.02)

        cube_now, cube_orn = p.getBasePositionAndOrientation(cube)
        ee_state = p.getLinkState(robot, 11, computeForwardKinematics=True)
        ee_pos, ee_orn = ee_state[0], ee_state[1]
        joint_angles = [p.getJointState(robot, i)[0] for i in range(7)]
        gripper = 1 if 30 <= step < 75 else 0
        rgb = capture_image()

        traj["frames"].append({
            "step": step,
            "object_pose": list(cube_now) + list(cube_orn),
            "ee_pose": list(ee_pos) + list(ee_orn),
            "joint_angles": joint_angles,
            "gripper_state": gripper
        })
        images.append(rgb)

        # 動態偵測抓取成功（只記錄一次）
        if step == 60:
            cube_tmp, _ = p.getBasePositionAndOrientation(cube)
            ee_tmp = p.getLinkState(robot, 11)[0]
            grasp_success = is_grasp_success(cube, ee_tmp)


    final_cube_pos, _ = p.getBasePositionAndOrientation(cube)
    # final_ee_pos = p.getLinkState(robot, 11)[0]
    # grasp_success = is_grasp_success(cube, final_ee_pos)
    task_success = is_task_success(final_cube_pos, box_pos)

    print("grasp_success: ", grasp_success)
    print("task_success: ", task_success)

    traj["grasp_success"] = str(grasp_success)
    traj["task_success"] = str(task_success)

    if grasp_success and task_success:
        save_trajectory(save_path, traj, images)
        print(f"[✓] Trajectory saved to {save_path} ✅ SUCCESS")
        p.disconnect()
        return True
    else:
        print("[✗] Trajectory not saved — failed attempt ❌")
        p.disconnect()
        return False

    # p.disconnect()

if __name__ == '__main__':
    record_one()
