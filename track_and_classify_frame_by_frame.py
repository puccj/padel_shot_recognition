"""
With this script you can infer a first basic shot detection/classification on a padel video.
For each frame, the movenet is inferred and we track player movement and pose.
Then, we feed this to the network trained in SingleFrameShotClassifier.ipynb
Finally, a pretty basic shot counter is applied to smooth shot probabilities over time
while preventing too close shots in terms of duration.
"""

from argparse import ArgumentParser, BooleanOptionalAction
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
from tqdm import tqdm

from extract_human_pose import HumanPoseExtractor, draw_pose
# from track_and_classify_with_rnn import GT, draw_probs
from track_and_classify_with_rnn import draw_probs

physical_devices = tf.config.experimental.list_physical_devices("GPU")
print(tf.config.experimental.list_physical_devices("GPU"))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


class ShotCounter:
    """Basic shot counter with a shot history"""

    BAR_WIDTH = 30
    BAR_HEIGHT = 170
    MARGIN_ABOVE_BAR = 30
    SPACE_BETWEEN_BARS = 55
    TEXT_ORIGIN_X = 1075
    BAR_ORIGIN_X = 1070

    def __init__(self, fps=60):
        self.nb_history = int(fps/2) # half a second history
        self.probs = np.zeros((self.nb_history, 4))

        self.nb_forehands = 0
        self.nb_backhands = 0
        self.nb_smashes = 0

        self.last_shot = "neutral"
        self.min_frames_between_shots = int(fps*2)  # 2 seconds between shots
        self.frames_since_last_shot = int(fps*2)  # TODO: maybe better to start with 0 (i.e. no shot for the first 2 seconds) ?

        self.results = []

    def update(self, probs, frame_id):
        """
        Update current state with new shots probabilities
        If one of the probability is over 50%, it can be considered as reliable
        We need at least min_frames_between_shots frames between two shots (backhand/forehand/smash)
        Between each shot, we should normally go through a "neutral state" meaning that the player
        is not currently hitting the ball
        """

        self.probs[0 : self.nb_history - 1, :] = self.probs[1:, :].copy()
        self.probs[-1, :] = probs

        self.frames_since_last_shot += 1

        means = np.mean(self.probs, axis=0)
        if means[0] > 0.5:          # backhand currently
            if (self.last_shot == "neutral"
                and self.frames_since_last_shot > self.min_frames_between_shots
            ):
                self.nb_backhands += 1
                self.last_shot = "backhand"
                self.frames_since_last_shot = 0
                self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif means[1] > 0.5:        # forehand currently
            if (self.last_shot == "neutral"
                and self.frames_since_last_shot > self.min_frames_between_shots
            ):
                self.nb_forehands += 1
                self.last_shot = "forehand"
                self.frames_since_last_shot = 0
                self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
        elif means[2] > 0.5:        # neutral currently
            self.last_shot = "neutral"
        elif means[3] > 0.5:        # smash currently
            if (self.last_shot == "neutral"
                and self.frames_since_last_shot > self.min_frames_between_shots
            ):
                self.nb_smashes += 1
                self.last_shot = "smash"
                self.frames_since_last_shot = 0
                self.results.append({"FrameID": frame_id, "Shot": self.last_shot})

    def display(self, frame):
        """
        Display shot count
        Colorize last shot in green
        """

        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (20, frame.shape[0] - 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color= (0, 255, 0) if (self.last_shot == "backhand" and self.frames_since_last_shot < 30) else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (20, frame.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color= (0, 255, 0) if (self.last_shot == "forehand" and self.frames_since_last_shot < 30) else (0, 0, 255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Smashes = {self.nb_smashes}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color= (0, 255, 0) if (self.last_shot == "smash" and self.frames_since_last_shot < 30) else (0, 0, 255),
            thickness=2,
        )


def compute_recall_precision(gt, shots):
    """
    Give some metrics to assess current performances, like
    how many shots were missed (recall) or were false positives (precision)
    """
    gt_numpy = gt.to_numpy()
    nb_match = 0
    nb_misses = 0
    nb_fp = 0
    fp_backhands = 0
    fp_forehands = 0
    fp_smashes = 0
    for gt_shot in gt_numpy:
        found_match = False
        for shot in shots:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if found_match:
            nb_match += 1
        else:
            nb_misses += 1

    for shot in shots:
        found_match = False
        for gt_shot in gt_numpy:
            if shot["Shot"] == gt_shot[0]:
                if abs(shot["FrameID"] - gt_shot[1]) <= 30:
                    found_match = True
                    break
        if not found_match:
            nb_fp += 1
            if shot["Shot"] == "backhand":
                fp_backhands += 1
            elif shot["Shot"] == "forehand":
                fp_forehands += 1
            elif shot["Shot"] == "smash":
                fp_smashes += 1

    precision = nb_match / (nb_match + nb_fp)
    recall = nb_match / (nb_match + nb_misses)

    print(f"Recall {recall*100:.1f}%")
    print(f"Precision {precision*100:.1f}%")

    print(f"FP: backhands = {fp_backhands}, forehands = {fp_forehands}, smash = {fp_smashes}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Track padel player and display shot probabilities")
    parser.add_argument("video")
    parser.add_argument("model")
    parser.add_argument("--evaluate", help="Path to annotation file")
    parser.add_argument("-s", "--show", action=BooleanOptionalAction, default=True, help='show video')
    parser.add_argument("-o", "--output", help="Path to output video file")

    args = parser.parse_args()

    # if args.evaluate is not None:
    #     gt = GT(args.evaluate)

    m1 = keras.models.load_model(args.model)
    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

    if not args.show:
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing", unit="frame")

    human_pose_extractor = HumanPoseExtractor(frame.shape)
    shot_counter = ShotCounter(fps)
    FRAME_ID = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        FRAME_ID += 1

        features = human_pose_extractor.extract(frame).reshape(17, 3)
        features_pixels = human_pose_extractor.transform_to_frame_coordinates(features)

        # Discard cancelled keypoints, flatten the array before feeding it to the model
        # and discard the confidence score
        features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

        probs = m1.__call__(features)[0] if human_pose_extractor.roi.valid else np.zeros(4)
        
        shot_counter.update(probs, FRAME_ID)

        draw_probs(frame, np.mean(shot_counter.probs, axis=0))
        shot_counter.display(frame)
        # draw_probs(frame, [probs[0], probs[1], probs[2], 0])

        # if args.evaluate is not None:
        #     gt.display(frame, FRAME_ID)
        
        if (
            shot_counter.frames_since_last_shot < 30
            and shot_counter.last_shot != "neutral"
        ):
            human_pose_extractor.roi.draw_shot(frame, shot_counter.last_shot)

        # Display results on original frame
        draw_pose(frame, features_pixels)
        human_pose_extractor.roi.update(features_pixels)

        if args.output is not None:
            out.write(frame)

        if args.show:
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == 27:  # ESC or 'q' to quit
                break
        else:
            pbar.update(1)


    
    cap.release()
    if args.output is not None:
        out.release()
    if args.show:
        cv2.destroyAllWindows()
    else:
        pbar.close()

    print(shot_counter.results)

    # if args.evaluate is not None:
    #     compute_recall_precision(gt.shots, shot_counter.results)
