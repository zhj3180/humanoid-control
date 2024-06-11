import math
import numpy as np

def inertia_matrix_to_quaternion(Ixx, Iyy, Izz, Ixy, Ixz, Iyz):
    """
    将惯性矩阵转换为四元数表示
    参数:
    Ixx, Iyy, Izz -- 惯性矩阵的对角线元素
    Ixy, Ixz, Iyz -- 惯性矩阵的非对角线元素
    
    返回:
    四元数（qx, qy, qz, qw）
    """
    I = np.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ])
    print("I: ", I)
    eigenvalues, eigenvectors = np.linalg.eig(I)
    
    # 要得到四元数，需要得到对应旋转矩阵R并进行转换
    R = eigenvectors  # R是用于对角化惯性矩阵的矩阵
    
    # 验证
    I_det = np.dot(R.transpose(), np.dot(I, R))
    print("I_det: ", I_det)

    # 从旋转矩阵到四元数
    qw = math.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    qx = (R[2,1] - R[1,2]) / (4.0 * qw)
    qy = (R[0,2] - R[2,0]) / (4.0 * qw)
    qz = (R[1,0] - R[0,1]) / (4.0 * qw)

    return qx, qy, qz, qw

# 示例使用
mass = 1
Ixx = 0.003430/mass
Ixy = 0.000980/mass
Ixz = -0.000017/mass
Iyy = 0.004095/mass
Iyz = -0.000026/mass
Izz = 0.004706/mass

qx, qy, qz, qw = inertia_matrix_to_quaternion(Ixx, Iyy, Izz, Ixy, Ixz, Iyz)
print(f"Quaternion: ({qw}, {qx}, {qy}, {qz})")