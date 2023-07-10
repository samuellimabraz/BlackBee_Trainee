import rospy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


class Detector:
    """
    Generic class to apply object detection in one ROS node
    with bebop or webcam image
    """

    def __init__(self, node_name: str):
        rospy.init_node(node_name, anonymous=True)

        self.image_topic = rospy.get_param("~image_topic", "webcam")
        print(f"Captura: {self.image_topic}")

        self.bridge = CvBridge()

    def detect(self, img):
        """
        Convert the image from ROS msg for bebop
        """
        if self.image_topic == "bebop":
            try:
                cv_img = self.bridge.compressed_imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)
        else:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
            except CvBridgeError as e:
                print(e)

        return cv_img

    def bebop_run(self):
        """
        Run detection with Bebop image topic
        """
        rospy.Subscriber(
            "bebop/image_raw/compressed", CompressedImage, self.detect, queue_size=10
        )
        rospy.spin()

    def webcam_run(self):
        """
        Run detection with webcam image
        """
        rospy.Subscriber('webcam_image', Image, self.detect)
        rospy.spin()

    def run(self):
        """
        Start node and select the image mode
        """
        if self.image_topic == "bebop":
            self.bebop_run()
        elif self.image_topic == "webcam":
            self.webcam_run()
