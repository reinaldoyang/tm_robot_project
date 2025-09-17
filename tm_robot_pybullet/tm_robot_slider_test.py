import pybullet as p
import pybullet_data
import time
import numpy as np
import math
from tm_robot_sim import create_simulation_env, attach_gripper_to_robot

def create_eef_sliders(name_prefix="eef"):
    """
    Creates sliders for end-effector XYZ position and roll/pitch/yaw in degrees.
    Returns a function to read current slider values as (x, y, z, roll, pitch, yaw) in radians.
    """
    # Position sliders
    x_slider = p.addUserDebugParameter(f"{name_prefix}_x", 0.0, 2.0, 0.6)
    y_slider = p.addUserDebugParameter(f"{name_prefix}_y", -1.0, 1.0, 0.0)
    z_slider = p.addUserDebugParameter(f"{name_prefix}_z", 0.0, 1.5, 1.1)
    
    # Orientation sliders (degrees)
    roll_slider  = p.addUserDebugParameter(f"{name_prefix}_roll", -180, 180, 0)
    pitch_slider = p.addUserDebugParameter(f"{name_prefix}_pitch", -180, 180, 180)
    yaw_slider   = p.addUserDebugParameter(f"{name_prefix}_yaw", -180, 180, 90)

    def read_sliders(radians=True):
        x = p.readUserDebugParameter(x_slider)
        y = p.readUserDebugParameter(y_slider)
        z = p.readUserDebugParameter(z_slider)
        roll = p.readUserDebugParameter(roll_slider)
        pitch = p.readUserDebugParameter(pitch_slider)
        yaw = p.readUserDebugParameter(yaw_slider)
        if radians:
            return x, y, z, math.radians(roll), math.radians(pitch), math.radians(yaw)
        else:
            return x, y, z, roll, pitch, yaw

    return read_sliders

def run_slider_test():
    # Create environment
    plane_id, robot_id, table_id, cube_id, tray_id, gripper_id = create_simulation_env("GUI")
    attach_gripper_to_robot(robot_id, gripper_id)
    
    end_effector_idx = 6  # TM robot flange
    read_eef = create_eef_sliders()
    
    while True:
        p.stepSimulation()
        time.sleep(1/240)
        
        # Read slider values
        x, y, z, roll, pitch, yaw = read_eef()
        target_pos = [x, y, z]
        target_orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        
        # Compute IK
        joint_poses = p.calculateInverseKinematics(robot_id, end_effector_idx, target_pos, target_orn)
        arm_joints = [1, 2, 3, 4, 5, 6]
        # Apply joint positions
        for i, j in enumerate(arm_joints):
            p.setJointMotorControl2(
                bodyIndex=robot_id,
                jointIndex=j,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=200,
                maxVelocity=3
            )
        
        # Optional: move gripper (fixed for now)
        gripper_val = 0
        p.setJointMotorControl2(gripper_id, 4, p.POSITION_CONTROL, targetPosition=gripper_val*0.05, force=100)
        p.setJointMotorControl2(gripper_id, 6, p.POSITION_CONTROL, targetPosition=gripper_val*0.05, force=100)
        
        # Optional: draw debug line to show gripper x-axis
        ee_pos, ee_orn = p.getLinkState(robot_id, end_effector_idx)[:2]
        ee_dir = p.getMatrixFromQuaternion(ee_orn)
        gripper_x = np.array([ee_dir[0], ee_dir[3], ee_dir[6]])  # x-axis in world frame
        p.addUserDebugLine(ee_pos, ee_pos + 0.2*gripper_x, [0,1,0], 3, lifeTime=0.1)
    

if __name__ == "__main__":
    run_slider_test()
    p.disconnect()

