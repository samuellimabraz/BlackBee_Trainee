import rospy
from std_msgs.msg import Empty, UInt8, Float32, Bool
from geometry_msgs.msg import Twist
from bebop_msgs.msg import Ardrone3PilotingStateAltitudeChanged


class Bebop:
    def __init__(self):
        # Declaração dos Publishers:
        self.takeoff_pub = rospy.Publisher("/bebop/takeoff", Empty, queue_size=1)
        self.land_pub = rospy.Publisher("/bebop/land", Empty, queue_size=1)
        self.vel_pub = rospy.Publisher("/bebop/cmd_vel", Twist, queue_size=1)
        self.flip_pub = rospy.Publisher("/bebop/flip", UInt8, queue_size=1)
        self.gimbal_pub = rospy.Publisher("/bebop/camera_control", Twist, queue_size=1)
        self.snap_pub = rospy.Publisher("/bebop/snapshot", Empty, queue_size=1)

        self.flying = False

    def takeoff(self):
        """
        Send command to take off and hold
        """
        self.flying = True
        self.takeoff_pub.publish()
        rospy.loginfo("-- Takeoff Started")

    def smooth_land(self, data):
        pass

    def land(self):
        """
        Send command to land at the current position.
        """

        # rospy.Subscriber('/bebop/states/ardrone3/PilotingState/AltitudeChanged',
        #                 Ardrone3PilotingStateAltitudeChanged, self.smooth_land)

        self.land_pub.publish()
        rospy.loginfo("-- Landing")

    def move_right(self):
        # Lógica para mover o drone para a direita
        pass

    def move_left(self):
        # Lógica para mover o drone para a esquerda
        pass

    def move_front(self):
        # Lógica para mover o drone para frente
        pass

    def flip(self):
        # Lógica para fazer o drone dar um flip
        pass

    def land(self):
        # Lógica para aterrissar o drone
        pass

    def wave(self):
        # Lógica para acenar com a mão
        pass
