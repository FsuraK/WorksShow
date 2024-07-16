import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import torch


class Subscriber(Node):
    def __init__(self, runner):
        super(Subscriber).__init__('converter')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'frame',
            self.callback,
            10)
        self.frame = None
        self.runner = runner
        self.subscription

    def callback(self, msg):
        array_2d = [msg.data[i:i+256] for i in range(0, len(msg.data), 256)]
        frame = torch.tensor(array_2d)  # convert the 2D list to a tensor
        self.frame = frame.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # add new dimensions
        self.runner.TimerCallBack(frame)


def main(args=None):
    rclpy.init(args=args)
    subscriber = Subscriber()
    rclpy.spin(subscriber)
    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
