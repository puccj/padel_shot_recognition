"""
Capture shots from annotation as a succession of features into a csv files
Note that we dont save useless features like eyes and ears positions.
"""

from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import os
from tqdm import tqdm

from extract_human_pose import (
    HumanPoseExtractor,
)

columns = [
    "nose_y",
    "nose_x",
    "left_shoulder_y",
    "left_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_x",
    "left_elbow_y",
    "left_elbow_x",
    "right_elbow_y",
    "right_elbow_x",
    "left_wrist_y",
    "left_wrist_x",
    "right_wrist_y",
    "right_wrist_x",
    "left_hip_y",
    "left_hip_x",
    "right_hip_y",
    "right_hip_x",
    "left_knee_y",
    "left_knee_x",
    "right_knee_y",
    "right_knee_x",
    "left_ankle_y",
    "left_ankle_x",
    "right_ankle_y",
    "right_ankle_x",
]


def draw_shot(frame, shot, side):
    """Draw shot name on frame (user-friendly)"""
    cv2.putText(
        frame,
        f"{side} {shot}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Annotate (associate human pose to a tennis shot)"
    )
    parser.add_argument("video")
    parser.add_argument("annotation")
    parser.add_argument("out")
    parser.add_argument("-s", "--show",  action=BooleanOptionalAction, help='show video frames')
    parser.add_argument("-d", "--debug", action=BooleanOptionalAction, help='show frame by frame and wait for key press')
    parser.add_argument("-v", "--verbose", action=BooleanOptionalAction, help='show verbose output')

    args = parser.parse_args()

    shots = pd.read_csv(args.annotation)
    CURRENT_ROW = 0

    shots_features = []

    FRAME_ID = 1    # frame number
    IDX_FOREHAND = 1
    IDX_BACKHAND = 1
    IDX_NEUTRAL = 1
    IDX_SMASH = 1

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {args.video}")

    # NB_IMAGES = 30  # number of images to capture for each shot
    NB_IMAGES = int(cap.get(cv2.CAP_PROP_FPS))   # 1 second of video

    ret, frame = cap.read()

    r_human_pose_extractor = HumanPoseExtractor(frame.shape, 'right', args.verbose)
    l_human_pose_extractor = HumanPoseExtractor(frame.shape, 'left', args.verbose)
    # model = YOLO("yolov8n.pt")

    os.makedirs(args.out, exist_ok=True)

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Progress bar
    if not args.verbose:
        pbar = tqdm(total=frames, desc="Progress", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        # bboxes = []
        # results = model(frame, verbose=False)[0]
        # for box in results.boxes:
        #     result = box.xyxy.tolist()[0]
        #     bboxes.append([int(result[0]), int(result[1]), int(result[2]), int(result[3])])

        if CURRENT_ROW >= len(shots):
            break

        r_human_pose_extractor.extract(frame)
        l_human_pose_extractor.extract(frame)

        # dont draw non-significant points/edges by setting probability to 0
        r_human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
        l_human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        if shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2 == FRAME_ID:
            shots_features = []

        shot_side = shots.iloc[CURRENT_ROW]["Player"]
        if shot_side == "right":
            human_pose_extractor = r_human_pose_extractor
        elif shot_side == "left":
            human_pose_extractor = l_human_pose_extractor
        elif shot_side == "one":
            # TODO: handle single player. For now, we consider the right player
            human_pose_extractor = r_human_pose_extractor
        else:
            raise ValueError(f"Unknown player side {shot_side}")
        
        features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
        confidence = np.mean(features[:, 2])

        if (    # if current frame is in the range of the current shot
            shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
            <= FRAME_ID
            <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
        ):
            if confidence < 0.3:
                if args.verbose:
                    print(f"Cancel {shots.iloc[CURRENT_ROW]['Shot']} shot, as not confident enough pose detected ({confidence:.2f})")
                CURRENT_ROW += 1
                shots_features = []
                FRAME_ID += 1
                continue

            features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)

            shot_class = shots.iloc[CURRENT_ROW]["Shot"]
            shots_features.append(features)
            if args.show:
                draw_shot(frame, shot_class, shot_side)

            if FRAME_ID - NB_IMAGES // 2 + 1 == shots.iloc[CURRENT_ROW]["FrameId"]:
                # add assert?
                shots_df = pd.DataFrame(
                    np.concatenate(shots_features, axis=0),
                    columns=columns,
                )
                shots_df["shot"] = np.full(NB_IMAGES, shot_class)
                if shot_class == "forehand":
                    outpath = Path(args.out).joinpath(
                        f"forehand_{IDX_FOREHAND:03d}.csv"
                    )
                    IDX_FOREHAND += 1
                elif shot_class == "backhand":
                    outpath = Path(args.out).joinpath(
                        f"backhand_{IDX_BACKHAND:03d}.csv"
                    )
                    IDX_BACKHAND += 1
                elif shot_class == "smash":
                    outpath = Path(args.out).joinpath(
                        f"smash_{IDX_SMASH:03d}.csv")
                    IDX_SMASH += 1

                shots_df.to_csv(outpath, index=False)

                assert len(shots_df) == NB_IMAGES

                if args.verbose:
                    print(f"saving {shot_class} to {outpath}")

                CURRENT_ROW += 1
                shots_features = []

        elif (  # if there is enough gap between current and previous shot, take a neutral shot
            shots.iloc[CURRENT_ROW]["FrameId"] - shots.iloc[CURRENT_ROW - 1]["FrameId"]
            > NB_IMAGES
        ):
            frame_id_between_shots = (
                shots.iloc[CURRENT_ROW - 1]["FrameId"]
                + shots.iloc[CURRENT_ROW]["FrameId"]
            ) // 2
            if (
                frame_id_between_shots - NB_IMAGES // 2
                < FRAME_ID
                <= frame_id_between_shots + NB_IMAGES // 2
            ):

                features = features[features[:, 2] > 0][:, 0:2].reshape(1, 13 * 2)
                shots_features.append(features)
                if args.show:
                    draw_shot(frame, "neutral", shot_side)

                if FRAME_ID == frame_id_between_shots + NB_IMAGES // 2:
                    shots_df = pd.DataFrame(
                        np.concatenate(shots_features, axis=0),
                        columns=columns,
                    )
                    shots_df["shot"] = np.full(NB_IMAGES, "neutral")
                    outpath = Path(args.out).joinpath(f"neutral_{IDX_NEUTRAL:03d}.csv")
                    if args.verbose:
                        print(f"saving neutral to {outpath}")
                    IDX_NEUTRAL += 1
                    shots_df.to_csv(outpath, index=False)
                    shots_features = []


        # Display results on original frame
        if args.show:
            r_human_pose_extractor.draw_results_frame(frame, (0, 0, 0), confidence)
            l_human_pose_extractor.draw_results_frame(frame, (0,255,255), confidence)
            # for bbox in bboxes:
            #     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, f"Frame {FRAME_ID}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)


        r_human_pose_extractor.roi.update(r_human_pose_extractor.keypoints_pixels_frame)
        l_human_pose_extractor.roi.update(l_human_pose_extractor.keypoints_pixels_frame)

        if args.show:
            if args.debug:
                k = cv2.waitKey(0)
            else:
                k = cv2.waitKey(1)
            
            if k == 27 or k == ord("q"):  # ESC or 'q' to quit
                break

        FRAME_ID += 1

        if not args.verbose:
            pbar.update(1)

    if not args.verbose:    
        pbar.close()
    cap.release()
    cv2.destroyAllWindows()

    print("Done, no more shots in annotation!")
    print(f"Extracted {IDX_FOREHAND - 1} forehands, {IDX_BACKHAND - 1} backhands, {IDX_SMASH - 1} smashes and {IDX_NEUTRAL - 1} neutrals")
    print(f"for a total of {IDX_FOREHAND + IDX_BACKHAND + IDX_SMASH - 3} shots out of {len(shots)} ({100 * (IDX_FOREHAND + IDX_BACKHAND + IDX_SMASH - 3) / len(shots):.2f}%)")
    print(f"If a small percentage of shots is extracted, consider changing the pose detection model or decreasing confidence threshold")
