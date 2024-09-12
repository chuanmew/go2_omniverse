from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Imu
import torch
import carb
from functools import partial

default_joint_states = torch.zeros((1,12), dtype=torch.float32, device='cuda')


def add_ros_obs_sub(ros_obs, node):

    def cmd_vel_callback(index: str):
        def callback(msg: Twist):
            x = msg.linear.x
            y = msg.linear.y
            z = msg.angular.z
            ros_obs[0, 9:12] = torch.tensor([x, y, z], device='cuda')
        return callback

    def joint_pos_callback(index: str):
        def callback(msg: JointState):
            ros_obs[0, 12:24] = torch.tensor(msg.position, device='cuda') - default_joint_states
        return callback

    def default_joint_pos_callback(index: str):
        def callback(msg: JointState):
            default_joint_states[:,:] = torch.tensor(msg.position, device='cuda')
        return callback

    def joint_vel_callback(index: str):
        def callback(msg: JointState):
            ros_obs[0, 24:36] = torch.tensor(msg.position, device='cuda')
        return callback

    def imu_callback(index: str):
        def callback(msg: JointState):
            ros_obs[0, 0:6] = torch.tensor([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
                                            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], device='cuda')
        return callback

    node.create_subscription(Twist, f'robot0/cmd_vel', cmd_vel_callback('0'), 10)
    node.create_subscription(JointState, f'robot0/joint_states', joint_pos_callback('0'), 10)
    node.create_subscription(JointState, f'robot0/default_joint_states', default_joint_pos_callback('0'), 10)
    node.create_subscription(JointState, f'robot0/joint_vel_states', joint_vel_callback('0'), 10)
    node.create_subscription(Imu, f'robot0/imu', imu_callback('0'), 10)


command = [0.0, 0.0, 0.0]


def sub_keyboard_event(event, publisher) -> bool:
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


def add_keyboard_subscription(_input, _keyboard, publisher):
    # 使用 partial 来创建带有额外参数的回调函数
    callback_with_args = partial(sub_keyboard_event, publisher=publisher)

    # 订阅键盘事件
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, callback_with_args)
