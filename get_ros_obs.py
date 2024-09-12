from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float32MultiArray
import torch
import carb


class ActionPublisherNode(Node):
    def __init__(self, obs):
        super().__init__('real_obs')
        self.default_joint_states = torch.zeros((1,12), dtype=torch.float32, device='cuda')
        self.ros_obs = torch.zeros((1,235), dtype=torch.float32, device='cuda')
        self.ros_obs = obs
        self.create_subscription(Twist, f'robot0/cmd_vel', self.cmd_vel_callback('0'), 10)
        self.create_subscription(JointState, f'robot0/joint_states', self.joint_pos_callback('0'), 10)
        self.create_subscription(JointState, f'robot0/default_joint_states', self.default_joint_pos_callback('0'), 10)
        self.create_subscription(JointState, f'robot0/joint_vel_states', self.joint_vel_callback('0'), 10)
        self.create_subscription(Imu, f'robot0/imu', self.imu_callback('0'), 10)
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

    def default_joint_pos_callback(self, index: str):
        def callback(msg: JointState):
            self.default_joint_states = torch.tensor(msg.position, device='cuda')
        return callback

    def joint_vel_callback(self, index: str):
        def callback(msg: JointState):
            self.ros_obs[0, 24:36] = torch.tensor(msg.position, device='cuda')
        return callback

    def imu_callback(self, index: str):
        def callback(msg: JointState):
            self.ros_obs[0, 0:6] = torch.tensor([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                                            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], device='cuda')
        return callback

    def generate_actions(self):
        indices = torch.tensor([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10])
        processed_actions_ordered = self.ros_obs[0, 12 + indices].tolist()
        action_msg = Float32MultiArray()
        action_msg.data = processed_actions_ordered
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
