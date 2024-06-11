import math

def deg_to_rad(deg):
    """Convert degrees to radians."""
    return deg * (math.pi / 180.0)

def rpy_to_quaternion(roll, pitch, yaw):
    """
    Convert RPY (roll, pitch, yaw) angles in degrees to quaternion.
    
    Args:
    roll  - Roll angle in degrees
    pitch - Pitch angle in degrees
    yaw   - Yaw angle in degrees

    Returns:
    (x, y, z, w) - Quaternion representation
    """
    # Convert degrees to radians
    # roll = deg_to_rad(roll)
    # pitch = deg_to_rad(pitch)
    # yaw = deg_to_rad(yaw)
    
    # Compute the quaternion
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w

# 示例

roll = -0.34907  # Roll角度，单位为度
pitch = 0.  # Pitch角度，单位为度
yaw = 0  # Yaw角度，单位为度

# 转换为四元数
quaternion = rpy_to_quaternion(roll, pitch, yaw)

# 输出结果
print(f"四元数表示: x={quaternion[0]}, y={quaternion[1]}, z={quaternion[2]}, w={quaternion[3]}")