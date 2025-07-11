"""
Read a video with opencv and infer movenet to display human pose.
Note that we perform tracking of the padel player to feed the neural network
with a more specific search area dennotated as RoI (Region of Interest).
If the player is lost, we reset the RoI.
"""

from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import cv2


class RoI:
    """
    Define the Region of Interest around the padel player.
    At each frame, we refine it and use current position to feed the
    movenet neural network.
    """

    def __init__(self, shape, side=None, verbose=False):
        self.side = side
        self.frame_width = shape[1]
        self.frame_height = shape[0]
        self.width = self.frame_width if side is None else self.frame_width // 2
        self.height = self.frame_height
        if side is None:
            self.center_x = shape[1] // 2
        elif side == "left":
            self.center_x = shape[1] // 4 
        elif side == "right":
            self.center_x = shape[1] * 3 // 4
        else:
            raise ValueError(f"Unknown side: {side}")
        self.center_y = shape[0] // 2
        self.valid = False
        self.verbose = verbose

    def extract_subframe(self, frame):
        """Extract the RoI from the original frame"""
        subframe = frame.copy()
        return subframe[
            self.center_y - self.height // 2 : self.center_y + self.height // 2,
            self.center_x - self.width // 2 : self.center_x + self.width // 2,
        ]

    def transform_to_subframe_coordinates(self, keypoints_from_tf):
        """Key points from tensorflow come as float number betwen 0 and 1,
        describing (x, y) coordinates in the image feeding the NN
        We transform them into sub frame pixel coordinates
        """
        return np.squeeze(
            np.multiply(keypoints_from_tf, [self.width, self.width, 1])
        ) - np.array([(self.width - self.height) // 2, 0, 0])

    def transform_to_frame_coordinates(self, keypoints_from_tf):
        """Key points from tensorflow come as float number betwen 0 and 1,
        describing (x, y) coordinates in the image feeding the NN
        We transform them into frame pixel coordinates
        """
        keypoints_pixels_subframe = self.transform_to_subframe_coordinates(
            keypoints_from_tf
        )
        keypoints_pixels_frame = keypoints_pixels_subframe.copy()
        keypoints_pixels_frame[:, 0] += self.center_y - self.height // 2
        keypoints_pixels_frame[:, 1] += self.center_x - self.width // 2

        return keypoints_pixels_frame

    def update(self, keypoints_pixels):
        """Update RoI with new keypoints.

        Parameters
        ----------
        keypoints_pixels_frame : np.ndarray
            An array of shape (17, 3) containing the keypoints in pixel coordinates.
            Each keypoint is represented by (y, x, confidence).
        """
        min_x = int(min(keypoints_pixels[:, 1]))
        min_y = int(min(keypoints_pixels[:, 0]))
        max_x = int(max(keypoints_pixels[:, 1]))
        max_y = int(max(keypoints_pixels[:, 0]))

        self.center_x = (min_x + max_x) // 2
        self.center_y = (min_y + max_y) // 2

        prob_mean = np.mean(keypoints_pixels[keypoints_pixels[:, 2] != 0][:, 2])
        if self.width != self.frame_width and prob_mean < 0.3:
            if self.verbose:
                print(f"Lost player track --> reset ROI because prob is too low = {prob_mean}")
            self.reset()
            return

        # keep next dimensions always a bit larger
        self.width = int((max_x - min_x) * 1.3)
        self.height = int((max_y - min_y) * 1.3)

        # TODO: make this dependent on the frame size
        if self.height < 90:
            if self.verbose:
                print(f"Reset ROI because height = {self.height}")
            self.reset()
            return
        # if self.width < 10:
        #     if self.verbose:
        #         print(f"Lost player track --> reset ROI because width = {self.width}")
        #     self.reset()
        #     return

        self.width = max(self.width, self.height)
        self.height = max(self.width, self.height)

        if self.center_x + self.width // 2 >= self.frame_width:
            self.center_x = self.frame_width - self.width // 2 - 1

        if 0 > self.center_x - self.width // 2:
            self.center_x = self.width // 2 + 1

        if self.center_y + self.height // 2 >= self.frame_height:
            self.center_y = self.frame_height - self.height // 2 - 2

        if 0 > self.center_y - self.height // 2:
            self.center_y = self.height // 2 + 1

        if self.side == 'right' and self.center_x < self.frame_width / 2:
            self.reset()
            if self.verbose:
                print("Right RoI is on the left side")
            return
        elif self.side == 'left' and self.center_x > self.frame_width / 2:
            self.reset()
            if self.verbose:
                print("Left RoI is on the right side")
            return

        # Reset if Out of Bound
        if self.center_x + self.width // 2 >= self.frame_width:
            self.reset()
            return

        # Reset if Out of Bound
        if self.center_y + self.height // 2 >= self.frame_height:
            self.reset()
            return

        assert 0 <= self.center_x - self.width // 2
        assert self.center_x + self.width // 2 < self.frame_width
        assert 0 <= self.center_y - self.height // 2
        assert self.center_y + self.height // 2 < self.frame_height

        # Set valid to True
        self.valid = True

    def reset(self):
        """
        Reset the RoI with width/height corresponding to the whole image
        """
        self.width = self.frame_width if self.side is None else self.frame_width // 2
        self.height = self.frame_height
        if self.side is None:
            self.center_x = self.frame_width // 2
        elif self.side == "left":
            self.center_x = self.frame_width // 4
        elif self.side == "right":
            self.center_x = self.frame_width * 3 // 4

        self.center_y = self.frame_height // 2

        self.valid = False

    def draw_shot(self, frame, shot):
        """Draw shot name in orange around bounding box"""
        cv2.putText(
            frame,
            shot,
            (self.center_x - 50, self.center_y - self.height // 2 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(128, 255, 255),
            thickness=2,
        )

EDGES = {
    (0, 1): "m",
    (0, 2): "c",
    (1, 3): "m",
    (2, 4): "c",
    (0, 5): "m",
    (0, 6): "c",
    (5, 7): "m",
    (7, 9): "m",
    (6, 8): "c",
    (8, 10): "c",
    (5, 6): "y",
    (5, 11): "m",
    (6, 12): "c",
    (11, 12): "y",
    (11, 13): "m",
    (13, 15): "m",
    (12, 14): "c",
    (14, 16): "c",
}

COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

class PoseExtractor:
    """Extract human pose from a video frame using the a pose estimation tflite model.
    
    For each frame, you are expected to call the `extract` method and update the RoI with features 
    extracted from the frame in frame/pixel coordinates. To convert the keypoints from normalized 
    (between 0 and 1) to pixel coordinates, you can use the `transform_to_frame_coordinates` method.
    
    """

    def __init__(self, shape, model_path="models/movenet_thunder.tflite", side=None, verbose=False):
        """Initialize the HumanPoseExtractor with the given RoI shape and side.

        Parameters
        ----------
        shape : tuple
            The shape of the RoI (height, width, channels) to be used for inference.
        model_path : str, optional
            The path to the TFLite model file. Default is "models/movenet_thunder.tflite".
            This model should be a pre-trained pose estimation tflite model.
        side : str, optional
            The side of the RoI, can be 'left', 'right', or None (default is None).
            If not None, only the left or right half of the frame will be used for inference.
        verbose : bool, optional
            If True, print additional information about the RoI and its updates.
            Default is False.
        """

        # Initialize the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.dimensions = self.interpreter.get_input_details()[0]["shape"]
        self.interpreter.allocate_tensors()

        self.roi = RoI(shape, side, verbose)

    def extract(self, frame):
        """Run inference model on subframe.
        
        Parameters
        ----------
        frame : np.ndarray
            The input frame from which to extract human pose.

        Returns
        -------
        keypoints : np.ndarray
            An array of shape (1, 1, 17, 3) containing the keypoints detected in the subframe.
            Each keypoint is represented by (y, x, confidence)
        """ 

        # Reshape image
        subframe = self.roi.extract_subframe(frame)

        img = subframe.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), self.dimensions[1], self.dimensions[2])
        input_image = tf.cast(img, dtype=tf.uint8)
        # input_image = tf.cast(img, dtype=tf.int32)

        # Setup input and output
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Make predictions
        self.interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(output_details[0]["index"])

        # Discard unnecessary keypoints (eyes or ears)
        indexes_to_discard = [KEYPOINT_DICT[k] for k in ["left_eye", "right_eye", "left_ear", "right_ear"]]
        keypoints[0, 0, indexes_to_discard, 2] = 0
        
        return keypoints
        # self.keypoints_pixels_frame = self.roi.transform_to_frame_coordinates(
        #     self.keypoints_with_scores
        # )
    
    def transform_to_frame_coordinates(self, keypoints_with_scores):
        return self.roi.transform_to_frame_coordinates(keypoints_with_scores)

    # def discard(self, list_of_keypoints):
    #     """Discard some points like eyes or ears (useless for our application)"""
    #     for keypoint in list_of_keypoints:
    #         self.keypoints_with_scores[0, 0, self.KEYPOINT_DICT[keypoint], 2] = 0
    #         self.keypoints_pixels_frame[self.KEYPOINT_DICT[keypoint], 2] = 0

    def draw_results_subframe(self, keypoints_with_scores):
        """Draw key points and eges on subframe (roi)"""
        subframe = self.roi.extract_subframe(frame)
        keypoints_pixels_subframe = self.roi.transform_to_subframe_coordinates(keypoints_with_scores)

        # Rendering
        draw_pose(subframe, keypoints_pixels_subframe, 0.2)

        return subframe

    def draw_roi(self, frame, color=(0, 255, 255), confidence=None):
        """Draw key points and eges on frame. Return True if the RoI is valid and drawn.
        
        Parameters
        ----------
        frame : np.ndarray
            The original frame on which to draw the results.
        color : tuple, optional
            The color of the rectangle to draw around the RoI. Default is (0, 255, 255) (yellow).
        confidence : float, optional
            The confidence score to display above the rectangle. If None, no text is displayed.
            Default is None.

        Returns
        -------
        drawn : bool
            True if the RoI is valid and drawn, False otherwise.
        
        """
        if not self.roi.valid:
            return False

        cx = self.roi.center_x
        cy = self.roi.center_y
        h = self.roi.height
        w = self.roi.width

        cv2.rectangle(frame, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), color, 3)

        if confidence:
            cv2.putText(
                frame,
                f"Confidence: {confidence:.2f}",
                (cx - w // 2, cy - h // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=color,
                thickness=2,
            )
        return True


class BBoxPoseExtractor:
    """Extract human pose from a video frame using the MoveNet model.
    
    This class is similar to HumanPoseExtractor, but it uses an external
    bounding box to extract the subframe instead of an internal RoI.
    The bounding box should be provided at each frame, in the extract method.
    """

    def __init__(self, verbose=False):
        """Initialize the BBoxHumanPoseExtractor with the given RoI shape and side.

        Parameters
        ----------
        verbose : bool, optional
            If True, print additional information about the RoI and its updates.
            Default is False.
        """

        # Initialize the TFLite interpreter
        self.interpreter = tf.lite.Interpreter(model_path="models/movenet_thunder.tflite")
        self.dimensions = self.interpreter.get_input_details()[0]["shape"]
        self.interpreter.allocate_tensors()
        self.verbose = verbose
    
    def extract(self, frame, enlarged_bbox):
        """Run inference model on subframe.

        Parameters
        ----------
        frame : np.ndarray
            The input frame from which to extract human pose.
        enlarged_bbox : tuple
            A tuple (x_min, y_min, x_max, y_max) representing the bounding box coordinates.
            Note that the bounding box should be enlarged to ensure that the pose is fully contained
            within the subframe. To do this, you can use the `enlarge_bounding_box` method.

        Returns
        -------
        keypoints : np.ndarray
            An array of shape (1, 1, 17, 3) containing the keypoints detected in the subframe.
            Each keypoint is represented by (y, x, confidence)
            If the bounding box is invalid, it returns None.
        """

        x1, y1, x2, y2 = enlarged_bbox
        h = y2 - y1

        # TODO: make this dependent on the frame size
        if h < 90:
            if self.verbose:
                print(f"Invalid bounding box because height = {h}")
            return None

        subframe = frame.copy()
        # img = subframe[big_y1:big_y2, big_x1:big_x2]
        img = subframe[y1:y2, x1:x2]
    
        # img = subframe.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), self.dimensions[1], self.dimensions[2])
        input_image = tf.cast(img, dtype=tf.uint8)
        # input_image = tf.cast(img, dtype=tf.int32)

        # Setup input and output
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Make predictions
        self.interpreter.set_tensor(input_details[0]["index"], np.array(input_image))
        self.interpreter.invoke()

        keypoints = self.interpreter.get_tensor(output_details[0]["index"])

        # Discard unnecessary keypoints (eyes or ears)
        indexes_to_discard = [KEYPOINT_DICT[k] for k in ["left_eye", "right_eye", "left_ear", "right_ear"]]
        keypoints[0, 0, indexes_to_discard, 2] = 0
        
        return keypoints    
    
    def transform_to_subframe_coordinates(self, keypoints_with_scores, enlarged_bbox):
        x1, y1, x2, y2 = enlarged_bbox
        width = x2 - x1
        height = y2 - y1
        # return np.squeeze(np.multiply(keypoints_with_scores, [width, width, 1])) - np.array([(width - height) // 2, 0, 0])
        result = keypoints_with_scores.copy()
        result[:, 0] *= height
        result[:, 1] *= width
        return result
    
    def transform_to_frame_coordinates(self, keypoints_with_scores, enlarged_bbox):
        """Transform keypoints from subframe coordinates to frame coordinates"""
        
        keypoints_pixels_subframe = self.transform_to_subframe_coordinates(keypoints_with_scores, enlarged_bbox)
        keypoints_pixels_frame = keypoints_pixels_subframe.copy()
        keypoints_pixels_frame[:, 0] += enlarged_bbox[1] # y1
        keypoints_pixels_frame[:, 1] += enlarged_bbox[0] # x1
        return keypoints_pixels_frame

    def draw_results_subframe(self, keypoints_with_scores, enlarged_bbox):
        """Draw key points and eges on subframe (bounding box). This method is usually 
        used to visualize the results of the pose estimation for debugging purposes."""
        x1, y1, x2, y2 = enlarged_bbox
        subframe = frame.copy()
        subframe = subframe[y1:y2, x1:x2]
        keypoints_pixels_subframe = self.transform_to_subframe_coordinates(keypoints_with_scores, enlarged_bbox)

        # Rendering
        draw_pose(subframe, keypoints_pixels_subframe, 0.2)

        return subframe

def enlarge_bounding_box(bounding_box):
    """Enlarge a given bounding box (x_min, y_min, x_max, y_max) by 30% and make it square."""

    x1, y1, x2, y2 = bounding_box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * 1.15
    h = (y2 - y1) * 1.15
    w = max(w, h)
    h = max(w, h)
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))

def draw_pose(frame, keypoints_pixels_frame, confidence_threshold=0.01):
    """Draw key points and eges on frame
    
    Parameters
    ----------
    frame : np.ndarray
        The original frame on which to draw the results.
    keypoints_pixels_frame : np.ndarray
        An array of shape (17, 3) containing the keypoints in pixel coordinates.
        Each keypoint is represented by (y, x, confidence).
    confidence_threshold : float, optional
        The minimum confidence threshold for drawing keypoints and edges.
        Default is 0.01.
    """

    # Draw edges
    for edge, color in EDGES.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints_pixels_frame[p1]
        y2, x2, c2 = keypoints_pixels_frame[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=COLORS[color],
                thickness=2,
            )

    # Draw keypoints
    for kp in keypoints_pixels_frame:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Display human pose on a video with HumanPose. " \
    "To see a usage example of BBoxPoseExtractor, check the multiple_people.py script.")
    parser.add_argument("video")
    parser.add_argument(
        "--debug",
        action="store_const",
        const=True,
        default=False,
        help="Show sub frame (RoI)",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    assert cap.isOpened()

    ret, frame = cap.read()

    human_pose_extractor = PoseExtractor(frame.shape)

    FRAME_ID = 0

    while cap.isOpened():
        ret, frame = cap.read()

        FRAME_ID += 1

        features = human_pose_extractor.extract(frame)

        # Extract subframe (roi) and display results
        if args.debug:
            subframe = human_pose_extractor.draw_results_subframe(features)
            confidence = np.mean(features[0, 0, :, 2])
            human_pose_extractor.draw_roi(frame, confidence=confidence)
            cv2.imshow("Subframe", subframe)

        # Display results on original frame
        features_frame = human_pose_extractor.transform_to_frame_coordinates(features)
        draw_pose(frame, features_frame)
        cv2.imshow("Frame", frame)
        human_pose_extractor.roi.update(features_frame)

        # cv2.imwrite(f"videos/image_{FRAME_ID:05d}.png", frame)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
