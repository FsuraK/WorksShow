from setuptools import setup, find_packages

setup(
    name="Radar",
    version="2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "rclpy",
        "std_msgs",
        "sensor_msgs"
    ],
    author="yy, Liu",
    description="A Radar target detection project using ROS2 and PyTorch.",
    license="MIT",
    keywords="ros2 pytorch radar",
)
