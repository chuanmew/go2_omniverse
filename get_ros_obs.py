from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32MultiArray
import torch
import carb
from rclpy.duration import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R
try:
    from unitree_go.msg import LowState, WirelessController
    print("Modules imported successfully.")
except ImportError as e:
    print(f"Error importing module: {e}")


class ActionPublisherNode(Node):
    def __init__(self):
        super().__init__('real_obs')
        self.twist = Twist()
        self.speed = 1.0  # Max lin speed, m/s
        self.turn = -1.0  # Max ang speed, rad/s (sign convention switch)
        self.last_msg_time = self.get_clock().now()
        self.timeout_duration = Duration(seconds=0.5)
        self.timer = self.create_timer(0.1, self.check_timeout)
        self.raw_actions = torch.zeros((1,12), dtype=torch.float32, device='cuda')
        self.default_joint_states = torch.tensor([0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5], dtype=torch.float32, device='cuda')
        self.ros_obs = torch.zeros((1,235), dtype=torch.float32, device='cuda')

        # for sim
        self.create_subscription(Twist, f'robot0/cmd_vel', self.cmd_vel_callback('0'), 10)
        self.create_subscription(JointState, f'robot0/joint_states', self.joint_pos_callback('0'), 10)
        self.create_subscription(JointState, f'robot0/joint_vel_states', self.joint_vel_callback('0'), 10)
        self.create_subscription(Imu, f'robot0/imu', self.imu_callback('0'), 10)

        # for real
        # self.create_subscription(LowState, '/lowstate', self.lowstate_callback, 10)
        # self.create_subscription(
        #     WirelessController,
        #     '/wirelesscontroller',
        #     self.wireless_controller_callback,
        #     10)

        self.create_subscription(Float32MultiArray, 'raw_actions', self.raw_actions_callback('0'), 10)
        self.publisher = self.create_publisher(Float32MultiArray, 'actions', 10)
        self.timer_period = 0.02  # 20 milliseconds
        self.start_delay = 5.0
        self.delay_timer = self.create_timer(self.start_delay, self.start_timer_callback)

    def start_timer_callback(self):
        # Create the actual timer
        self.create_timer(self.timer_period, self.generate_actions)
        # Destroy the delayed start timer
        self.destroy_timer(self.delay_timer)

    def cmd_vel_callback(self, index: str):
        def callback(msg: Twist):
            x = msg.linear.x
            y = msg.linear.y
            z = msg.angular.z
            self.ros_obs[0, 9:12] = torch.tensor([x, y, z], device='cuda')
        return callback

    def joint_pos_callback(self, index: str):
        def callback(msg: JointState):
            self.ros_obs[0, 12:24] = torch.tensor(msg.position, device='cuda') - self.default_joint_states
        return callback

    def joint_vel_callback(self, index: str):
        def callback(msg: JointState):
            self.ros_obs[0, 24:36] = torch.tensor(msg.position, device='cuda')
        return callback

    def imu_callback(self, index: str):
        def callback(msg: Imu):
            self.ros_obs[0, 0:6] = torch.tensor([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                                            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], device='cuda')
        return callback

    def lowstate_callback(self, msg):
        motor_qs = [
            msg.motor_state[3].q,  # 0  -> FL_hip_joint   to FL_hip   -> 3
            msg.motor_state[0].q,  # 1  -> FR_hip_joint   to FR_hip   -> 0
            msg.motor_state[9].q,  # 2  -> RL_hip_joint   to RL_hip   -> 9
            msg.motor_state[6].q,  # 3  -> RR_hip_joint   to RR_hip   -> 6
            msg.motor_state[4].q,  # 4  -> FL_thigh_joint to FL_thigh -> 4
            msg.motor_state[1].q,  # 5  -> FR_thigh_joint to FR_thigh -> 1
            msg.motor_state[10].q,  # 6  -> RL_thigh_joint to RL_thigh -> 10
            msg.motor_state[7].q,  # 7  -> RR_thigh_joint to RR_thigh -> 7
            msg.motor_state[5].q,  # 8  -> FL_calf_joint  to FL_calf  -> 5
            msg.motor_state[2].q,  # 9  -> FR_calf_joint  to FR_calf  -> 2
            msg.motor_state[11].q,  # 10 -> RL_calf_joint  to RL_calf  -> 11
            msg.motor_state[8].q,  # 11 -> RR_calf_joint  to RR_calf  -> 8
        ]
        motor_dqs = [
            msg.motor_state[3].dq,  # 0  -> FL_hip_joint   to FL_hip   -> 3
            msg.motor_state[0].dq,  # 1  -> FR_hip_joint   to FR_hip   -> 0
            msg.motor_state[9].dq,  # 2  -> RL_hip_joint   to RL_hip   -> 9
            msg.motor_state[6].dq,  # 3  -> RR_hip_joint   to RR_hip   -> 6
            msg.motor_state[4].dq,  # 4  -> FL_thigh_joint to FL_thigh -> 4
            msg.motor_state[1].dq,  # 5  -> FR_thigh_joint to FR_thigh -> 1
            msg.motor_state[10].dq,  # 6  -> RL_thigh_joint to RL_thigh -> 10
            msg.motor_state[7].dq,  # 7  -> RR_thigh_joint to RR_thigh -> 7
            msg.motor_state[5].dq,  # 8  -> FL_calf_joint  to FL_calf  -> 5
            msg.motor_state[2].dq,  # 9  -> FR_calf_joint  to FR_calf  -> 2
            msg.motor_state[11].dq,  # 10 -> RL_calf_joint  to RL_calf  -> 11
            msg.motor_state[8].dq,  # 11 -> RR_calf_joint  to RR_calf  -> 8
        ]
        imu_quaternion = np.array([
            msg.imu_state.quaternion[0],
            msg.imu_state.quaternion[1],
            msg.imu_state.quaternion[2],
            msg.imu_state.quaternion[3]
        ])
        self.ros_obs[0, 6:9] = self.projected_gravity_vector(imu_quaternion)
        self.ros_obs[0, 0:6] = torch.tensor([msg.imu_state.accelerometer[0], msg.imu_state.accelerometer[1], msg.imu_state.accelerometer[2],
                                            msg.imu_state.gyroscope[0], msg.imu_state.gyroscope[1], msg.imu_state.gyroscope[2]], device='cuda')
        self.ros_obs[0, 12:24] = torch.tensor(motor_qs, device='cuda') - self.default_joint_states
        self.ros_obs[0, 24:36] = torch.tensor(motor_dqs, device='cuda')

    def projected_gravity_vector(self, imu_quaternion):
        # Use rotation from quaternion to find proj g
        rotation = R.from_quat(imu_quaternion)
        gravity_vec_w = np.array([0.0, 0.0, -1.0])  # Gravity vector in world
        gravity_proj = -1 * rotation.apply(gravity_vec_w)
        return gravity_proj

    def wireless_controller_callback(self, msg):
        self.twist = Twist()
        self.twist.linear.x = msg.ly * self.speed
        self.twist.linear.y = msg.lx * -self.speed  # (sign convention switch)
        self.twist.angular.z = msg.rx * self.turn
        self.ros_obs[0, 9:12] = torch.tensor([self.twist.linear.x, self.twist.linear.y, self.twist.linear.z], device='cuda')
        # Update the last message time
        self.last_msg_time = self.get_clock().now()

    def check_timeout(self):
        if self.get_clock().now() - self.last_msg_time > self.timeout_duration:
            zero_twist = Twist()
            self.ros_obs[0, 9:12] = torch.tensor([zero_twist.linear.x, zero_twist.linear.y, zero_twist.linear.z], device='cuda')

    def raw_actions_callback(self, index: str):
        def callback(msg: Float32MultiArray):
            self.raw_actions = torch.tensor(msg.data, device='cuda')
            self.ros_obs[0, 36:48] = self.raw_actions
        return callback

    def generate_actions(self):
        indices = torch.tensor([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10])
        action_tensor =  self.raw_actions * 0.5 + self.default_joint_states
        action_msg = Float32MultiArray()
        action_msg.data = action_tensor[indices].tolist()
        self.publisher.publish(action_msg)


command = [0.0, 0.0, 0.0]


def add_keyboard_subscription(_input, _keyboard, publisher):
    def sub_keyboard_event(event, publisher=publisher) -> bool:
        global command
        msg = Twist()
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                command = [1.0, 0.0, 0.0]
            if event.input.name == 'S':
                command = [-1.0, 0.0, 0.0]
            if event.input.name == 'A':
                command = [0.0, 1.0, 0.0]
            if event.input.name == 'D':
                command = [0.0, -1.0, 0.0]
            if event.input.name == 'Q':
                command = [0.0, 0.0, 1.0]
            if event.input.name == 'E':
                command = [0.0, 0.0, -1.0]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            command = [0.0, 0.0, 0.0]
        msg.linear.x = command[0]
        msg.linear.y = command[1]
        msg.angular.z = command[2]
        publisher.publish(msg)
        return True

    # 订阅键盘事件
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)
