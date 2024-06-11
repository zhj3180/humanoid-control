import math

def quaternion_to_euler(x, y, z, w):
    """
    将四元数 (x, y, z, w) 转换为欧拉角 (roll, pitch, yaw)
    
    参数:
    x, y, z, w -- 四元数的分量
    
    返回:
    roll, pitch, yaw -- 欧拉角（单位：弧度）
    
    """
    stand = 180/math.pi
    stand=1
    # 计算 roll
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)/stand

    # 计算 pitch
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)/stand
    # 计算 yaw
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)/stand

    return roll, pitch, yaw

# 示例使用
q_w, q_x, q_y, q_z = 0, -0.382683, 0 ,0.92388
roll, pitch, yaw = quaternion_to_euler(q_x, q_y, q_z, q_w)

print("Roll: ", math.degrees(roll), "度")
print("Pitch: ", math.degrees(pitch), "度")
print("Yaw: ", math.degrees(yaw), "度")