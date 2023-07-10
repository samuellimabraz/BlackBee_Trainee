import rospy
from std_msgs.msg import Empty, UInt8, Float32, Bool
from geometry_msgs.msg import Twist
from bebop_msgs.msg import Ardrone3PilotingStateAltitudeChanged

import numpy as np


class Bebop:
    def __init__(self):
        # Declaração dos Publishers:
        self.takeoff_pub = rospy.Publisher("/bebop/takeoff", Empty, queue_size=1)
        self.land_pub = rospy.Publisher("/bebop/land", Empty, queue_size=1)
        self.vel_pub = rospy.Publisher("/bebop/cmd_vel", Twist, queue_size=1)
        self.flip_pub = rospy.Publisher("/bebop/flip", UInt8, queue_size=1)
        self.gimbal_pub = rospy.Publisher("/bebop/camera_control", Twist, queue_size=1)
        self.snap_pub = rospy.Publisher("/bebop/snapshot", Empty, queue_size=1)

        self.altitude = 0.0
        rospy.Subscriber(
            "/bebop/states/ardrone3/PilotingState/AltitudeChanged",
            Ardrone3PilotingStateAltitudeChanged,
            lambda msg: setattr(self, "altitude", msg.altitude),
        )

        self.flying = False

    def takeoff(self):
        """
        Send command to take off and hold
        """
        self.flying = True
        self.takeoff_pub.publish()
        rospy.loginfo("-- Takeoff Started")

    def offboard_velocity(
        self,
        linear_x: float = 0.0,
        linear_y: float = 0.0,
        linear_z: float = 0.0,
        angular_z: float = 0.0,
    ):
        """
        Move sending velocity commands

        Acceptable range for all fields are [-1..1]

        Parameters
        ----------
        linear_x: float
            (+)Move forward

            (-)Move backward

        linear_y: float
            (+)Move left

            (-)Move right

        linear_z: float
            (+)Move up

            (-)Move down

        angular_z: float
            (+)Rotate counter clockwise

            (-)Rotate clockwise
        """
        params = [linear_x, linear_y, linear_z, angular_z]
        val_array = np.clip(params, -1, 1)

        vel_msg = Twist()
        vel_msg.linear.x = val_array[0]
        vel_msg.linear.y = val_array[1]
        vel_msg.linear.z = val_array[2]
        vel_msg.angular.z = val_array[3]

        self.vel_pub.publish(vel_msg)

    def land(self):
        """
        Send command to land at the current position.
        """

        self.land_pub.publish()
        rospy.loginfo("-- Landing")

    def wave(self, ang: float):
        ang = np.clip(ang, -1, 1)
        print("Wave moviment")
        for _ in range(3):
            t_start = t_now = rospy.get_rostime()
            duration = rospy.Duration(1)

            while t_now <= t_start + duration:
                self.offboard_velocity(angular_z=ang)
                t_now = rospy.get_rostime()

            t_start = t_now = rospy.get_rostime()
            while t_now <= t_start + (2 * duration):
                self.offboard_velocity(angular_z=-ang)
                t_now = rospy.get_rostime()

            t_start = t_now = rospy.get_rostime()
            while t_now <= t_start + duration:
                self.offboard_velocity(angular_z=ang)
                t_now = rospy.get_rostime()

    def screw(self):
        print("Screw Moviment")
        t_start = t_now = rospy.get_rostime()
        duration = rospy.Duration(5)

        while t_now <= t_start + duration:
            if self.altitude > 4:
                self.offboard_velocity(linear_z=0.0, angular_z=0.0)
            self.offboard_velocity(linear_z=0.1, angular_z=0.314)
            t_now = rospy.get_rostime()

        t_start = rospy.get_rostime()
        while t_now <= t_start + duration:
            if self.altitude < 2:
                self.offboard_velocity(linear_z=0.0, angular_z=0.0)
            self.offboard_velocity(linear_z=-0.1, angular_z=-0.314)
            t_now = rospy.get_rostime()
