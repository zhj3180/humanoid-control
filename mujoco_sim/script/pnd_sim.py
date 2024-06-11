#!/usr/bin/env python3
import mujoco as mj
import numpy as np
from mujoco_base import MuJoCoBase
from mujoco.glfw import glfw
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray,Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import time

from geometry_msgs.msg import Vector3


class HumanoidSim(MuJoCoBase):
  def __init__(self, xml_path):
    super().__init__(xml_path)
    self.simend = 1000.0
    self.sim_rate = 1000.0
    # print('Total number of DoFs in the model:', self.model.nv)
    # print('Generalized positions:', self.data.qpos)  
    # print('Generalized velocities:', self.data.qvel)
    # print('Actuator forces:', self.data.qfrc_actuator)
    # print('Actoator controls:', self.data.ctrl)
    # mj.set_mjcb_control(self.controller)
    # * Set subscriber and publisher


    ##############################机身全部质量，通过humanoidcontrol里的订阅函数发布，由pinocchio计算得到 64.38415

    # initialize target joint position, velocity, and torque
    
    # self.targetPos = np.array([0.05, -0.05, 0.35, -0.90, -0.55, 0, -0.05, 0.05, 0.35, -0.90, -0.55, 0])
    self.targetPos = np.zeros(12)
    self.targetVel = np.zeros(12)
    
    self.targetTorque = np.zeros(12)
    #self.targetTorque = np.zeros(13)    
    
    self.targetKp = np.ones(12) * 30    #30
    self.targetKd = np.ones(12) * 2

    self.pubJoints = rospy.Publisher('/jointsPosVel', Float32MultiArray, queue_size=10)
    self.pubOdom = rospy.Publisher('/ground_truth/state', Odometry, queue_size=10)
    self.pubImu = rospy.Publisher('/imu', Imu, queue_size=10)
    self.pubRealTorque = rospy.Publisher('/realTorque', Float32MultiArray, queue_size=10)
    
    self.disturbForce = Vector3()
    rospy.Subscriber("/disturb_force", Vector3, self.disturbForceCallback)

    rospy.Subscriber("/targetTorque", Float32MultiArray, self.targetTorqueCallback) 
    rospy.Subscriber("/targetPos", Float32MultiArray, self.targetPosCallback) 
    rospy.Subscriber("/targetVel", Float32MultiArray, self.targetVelCallback)
    rospy.Subscriber("/targetKp", Float32MultiArray, self.targetKpCallback)
    rospy.Subscriber("/targetKd", Float32MultiArray, self.targetKdCallback)
    #set the initial joint position
    # self.data.qpos[:3] = np.array([0, 0, 0.976])    #质心
    #修改
    self.data.qpos[:3] = np.array([0, 0, 0.925])    #质心
    # self.data.qpos[-12:] = np.array([0.05, -0.05, 0.35, -0.90, -0.55, 0, -0.05, 0.05, 0.35, -0.90, -0.55, 0])
    
    #qpos xyz-3/wxyz-4/腿-12/腰-3/手臂-12  34
    #qpos xyz-3/rby-3/腿-12/腰-3/手臂-12  33
    
    self.data.qpos[-27:-15] = np.zeros(12)    #腿部
    # self.data.qpos[-27:-15] = np.array([0.05, -0.05, 0.35, -0.90, -0.55, 0, -0.05, 0.05, 0.35, -0.90, -0.55, 0]) 
    self.data.qpos[-15:-12] = np.zeros(3)     #腰
    self.data.qpos[-12:] = np.zeros(12)       #手臂
    self.data.qvel[:3] = np.array([0, 0, 0])
    self.data.qvel[-27:-15] = np.zeros(12)    #腿部
    self.data.qvel[-15:-12] = np.zeros(3)     #腰
    self.data.qvel[-12:] = np.zeros(12)       #手臂
    

    # print(self.data.qpos)    
    # print(self.data.qvel)
    
    filename = "ctrl_data.txt"
        # 打开文件准备写入，模式为'append'
    with open(filename, "w") as file:
        # 取self.data.ctrl的前12个元素并转换成字符串列表
        # 然后使用join方法将它们连成一行，以空格分隔
        # 将这个字符串写入文件
        line = '\n'
        file.write(line)
        
    # filename = "ctrl_data_2.txt"
    #     # 打开文件准备写入，模式为'append'
    # with open(filename, "w") as file:
    #     # 取self.data.ctrl的前12个元素并转换成字符串列表
    #     # 然后使用join方法将它们连成一行，以空格分隔
    #     line4 = 'pos_data '+' '.join(map(str, self.data.qpos)) + '\n'
    #     file.write(line4)
    #     line6 = 'vel_data '+' '.join(map(str, self.data.qvel )) + '\n'
    #     file.write(line6)
    

    # * show the model
    mj.mj_step(self.model, self.data)
    # enable contact force visualization
    self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        self.window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    # Update scene and render
    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
    mj.mjr_render(viewport, self.scene, self.context)
    

  def targetTorqueCallback(self, data):
    self.targetTorque = data.data

  def targetPosCallback(self, data):
    self.targetPos = data.data

  def targetVelCallback(self, data):
    self.targetVel = data.data 

  def targetKpCallback(self, data):
    self.targetKp = data.data

  def targetKdCallback(self, data):
    self.targetKd = data.data
    
  def disturbForceCallback(self, data):
    self.disturbForce = data
    # if self.disturbForce.x or self.disturbForce.y or self.disturbForce.z:
    #   rospy.loginfo("PYTHON:::disturbForce: x=%f, y=%f, z=%f", self.disturbForce.x, self.disturbForce.y,self.disturbForce.z)

  def reset(self):
    # Set camera configuration
    self.cam.azimuth = 89.608063
    self.cam.elevation = -11.588379
    self.cam.distance = 5.0
    self.cam.lookat = np.array([0.0, 0.0, 1.5])

  # def controller(self, model, data):
  #   self.data.ctrl[0] = 100
  #   pass

  def simulate(self):
    publish_time = self.data.time
    torque_publish_time = self.data.time
    sim_epoch_start = time.time()
    while not glfw.window_should_close(self.window):
      simstart = self.data.time

      while (self.data.time - simstart <= 1.0/60.0 and not self.pause_flag):

        # MIT control
        
        #self.data.ctrl[:] = self.targetTorque + self.targetKp * (self.targetPos - self.data.qpos[-12:]) + self.targetKd * (self.targetVel - self.data.qvel[-12:])
        #修改
        # self.data.ctrl[0:12] = self.targetTorque + self.targetKp * (self.targetPos - self.data.qpos[-12:]) + self.targetKd * (self.targetVel - self.data.qvel[-12:])
        self.data.ctrl[0:12] = self.targetTorque + self.targetKp * (self.targetPos - self.data.qpos[-27:-15]) + self.targetKd * (self.targetVel - self.data.qvel[-27:-15])
        #self.data.ctrl[0:12] = self.targetTorque
    
        
        # self.data.ctrl[0:12] = np.array([-11.438448238391704 ,8.20674305385168 ,1.6617636624505947 ,48.246957006863006 ,3.9053953284753775 ,0.47210264880056896 ,
        #                                  15.2312221952863 ,-11.475300653214173 ,0.5448624504387695 ,41.1861825441725 ,0.7939015798768563 ,-1.042886952502611])
        
        # self.data.ctrl[0:12] = np.array([-11.438448238391704 ,8.20674305385168 ,1.6617636624505947 ,14.246957006863006 ,25.9053953284753775 ,25.47210264880056896 ,
        #                                  15.2312221952863 ,-11.475300653214173 ,0.5448624504387695 ,14.1861825441725 ,25.7939015798768563 ,25.042886952502611])
        
        filename = "ctrl_data.txt"
        # 打开文件准备写入，模式为'append'
        with open(filename, "a") as file:
            # 取self.data.ctrl的前12个元素并转换成字符串列表
            # 然后使用join方法将它们连成一行，以空格分隔
            line1 = 'torque_all '+' '.join(map(str, self.data.ctrl[0:12])) + '\n'
            file.write(line1)
            line2 = 'torque_target '+' '.join(map(str, self.targetTorque)) + '\n'
            file.write(line2)
            line3 = 'pos_target '+' '.join(map(str, self.targetPos)) + '\n'
            file.write(line3)
            line4 = 'pos_data '+' '.join(map(str, self.data.qpos[-27:-15])) + '\n'
            file.write(line4)
            line5 = 'vel_target '+' '.join(map(str, self.targetVel)) + '\n'
            file.write(line5)
            line6 = 'vel_data '+' '.join(map(str, self.data.qvel[-27:-15] )) + '\n'
            file.write(line6)
            line7 = 'pos_diff '+' '.join(map(str, self.targetPos - self.data.qpos[-27:-15])) + '\n'
            file.write(line7)
            line8 = 'vel_diff '+' '.join(map(str, self.targetVel - self.data.qvel[-27:-15])) + '\n'
            file.write(line8)
            
        


#没用             
#         num_bod = self.model.nbody
# # 使用 mjlib.mj_name2id 来获取 "base_link" 刚体的 ID
#         # body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base_link")
#         body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "leg_r6_link")
#         self.data.cdof[body_id] = np.array([self.disturbForce.x, self.disturbForce.y, self.disturbForce.z,0,0,0]) 
        
        #rospy.loginfo("PYTHON:::disturbForce: x=%f, y=%f, z=%f", self.disturbForce.x, self.disturbForce.y,self.disturbForce.z)
        # Step simulation environment
        mj.mj_step(self.model, self.data)

        if (self.data.time - publish_time >= 1.0 / 500.0):
          # * Publish joint positions and velocities
          jointsPosVel = Float32MultiArray()
          # get last 12 element of qpos and qvel
          #修改
          qp = self.data.qpos[-27:-15].copy()
          qv = self.data.qvel[-27:-15].copy()
          jointsPosVel.data = np.concatenate((qp,qv))

          self.pubJoints.publish(jointsPosVel)
          # * Publish body pose
          bodyOdom = Odometry()
          pos = self.data.sensor('BodyPos').data.copy()
          ori = self.data.sensor('BodyQuat').data.copy()
          vel = self.data.qvel[:3].copy()
          angVel = self.data.sensor('BodyGyro').data.copy()

          bodyOdom.header.stamp = rospy.Time.now()
          bodyOdom.pose.pose.position.x = pos[0]
          bodyOdom.pose.pose.position.y = pos[1]
          bodyOdom.pose.pose.position.z = pos[2]
          bodyOdom.pose.pose.orientation.x = ori[1]
          bodyOdom.pose.pose.orientation.y = ori[2]
          bodyOdom.pose.pose.orientation.z = ori[3]
          bodyOdom.pose.pose.orientation.w = ori[0]
          bodyOdom.twist.twist.linear.x = vel[0]
          bodyOdom.twist.twist.linear.y = vel[1]
          bodyOdom.twist.twist.linear.z = vel[2]
          bodyOdom.twist.twist.angular.x = angVel[0]
          bodyOdom.twist.twist.angular.y = angVel[1]
          bodyOdom.twist.twist.angular.z = angVel[2]
          self.pubOdom.publish(bodyOdom)

          bodyImu = Imu()
          acc = self.data.sensor('BodyAcc').data.copy()
          bodyImu.header.stamp = rospy.Time.now()
          bodyImu.angular_velocity.x = angVel[0]
          bodyImu.angular_velocity.y = angVel[1]
          bodyImu.angular_velocity.z = angVel[2]
          bodyImu.linear_acceleration.x = acc[0]
          bodyImu.linear_acceleration.y = acc[1]
          bodyImu.linear_acceleration.z = acc[2]
          bodyImu.orientation.x = ori[1]
          bodyImu.orientation.y = ori[2]
          bodyImu.orientation.z = ori[3]
          bodyImu.orientation.w = ori[0]
          bodyImu.orientation_covariance = [0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0]
          bodyImu.angular_velocity_covariance = [0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0]
          bodyImu.linear_acceleration_covariance = [0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0]
          self.pubImu.publish(bodyImu)
          
          #print("acc:::",bodyImu.linear_acceleration.x,bodyImu.linear_acceleration.y,bodyImu.linear_acceleration.z)

          publish_time = self.data.time

      if (self.data.time - torque_publish_time >= 1.0 / 40.0):
        targetTorque = Float32MultiArray()
        targetTorque.data = self.data.ctrl[:]
        self.pubRealTorque.publish(targetTorque)
        torque_publish_time = self.data.time

      if self.data.time >= self.simend:
          break
      if self.pause_flag:
        # publish the state even if the simulation is paused
        # * Publish joint positions and velocities
        jointsPosVel = Float32MultiArray()
        # get last 12 element of qpos and qvel
        #修改
        qp = self.data.qpos[-27:-15].copy()
        qv = np.zeros(12)
        jointsPosVel.data = np.concatenate((qp,qv))

        self.pubJoints.publish(jointsPosVel)
        # * Publish body pose
        bodyOdom = Odometry()
        pos = self.data.sensor('BodyPos').data.copy()
        ori = self.data.sensor('BodyQuat').data.copy()
        vel = self.data.qvel[:3].copy()
        angVel = self.data.sensor('BodyGyro').data.copy()
        bodyOdom.header.stamp = rospy.Time.now()
        bodyOdom.pose.pose.position.x = pos[0]
        bodyOdom.pose.pose.position.y = pos[1]
        bodyOdom.pose.pose.position.z = pos[2]
        bodyOdom.pose.pose.orientation.x = ori[1]
        bodyOdom.pose.pose.orientation.y = ori[2]
        bodyOdom.pose.pose.orientation.z = ori[3]
        bodyOdom.pose.pose.orientation.w = ori[0]
        bodyOdom.twist.twist.linear.x = 0
        bodyOdom.twist.twist.linear.y = 0
        bodyOdom.twist.twist.linear.z = 0
        bodyOdom.twist.twist.angular.x = 0
        bodyOdom.twist.twist.angular.y = 0
        bodyOdom.twist.twist.angular.z = 0
        self.pubOdom.publish(bodyOdom)

        bodyImu = Imu()
        bodyImu.header.stamp = rospy.Time.now()
        bodyImu.angular_velocity.x = 0
        bodyImu.angular_velocity.y = 0
        bodyImu.angular_velocity.z = 0
        bodyImu.linear_acceleration.x = 0
        bodyImu.linear_acceleration.y = 0
        bodyImu.linear_acceleration.z = 9.81
        bodyImu.orientation.x = ori[1]
        bodyImu.orientation.y = ori[2]
        bodyImu.orientation.z = ori[3]
        bodyImu.orientation.w = ori[0]
        self.pubImu.publish(bodyImu)

      # get framebuffer viewport
      viewport_width, viewport_height = glfw.get_framebuffer_size(
          self.window)
      viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

      # Update scene and render
      mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                          mj.mjtCatBit.mjCAT_ALL.value, self.scene)
      mj.mjr_render(viewport, self.scene, self.context)

      # swap OpenGL buffers (blocking call due to v-sync)
      glfw.swap_buffers(self.window)

      # process pending GUI events, call GLFW callbacks
      glfw.poll_events()

    glfw.terminate()

def main():
    # ros init
    rospy.init_node('hector_sim', anonymous=True)

    # get xml path
    rospack = rospkg.RosPack()
    rospack.list()
    # hector_desc_path = rospack.get_path('humanoid_legged_description')
    # xml_path = hector_desc_path + "/mjcf/humanoid_legged.xml"
    xml_path = "/home/zhanghj/software/human_ws/src/humanoid-control/adam_lite/urdf/adam_lite_copy.xml"
    sim = HumanoidSim(xml_path)
    sim.reset()
    sim.simulate()

if __name__ == "__main__":
    main()
