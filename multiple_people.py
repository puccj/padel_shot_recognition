"""Track and classify the shots of multiple people in a video frame by frame.
This script utilizes an object detection model (e.g. YOLO) to track multiple players, the HumanPoseExtractor class to extract their
poses, and the shot classification model trained in SingleFrameShotClassifier.ipynb to classify their shots.
It also implements a basic shot counter to smooth shot probabilities over time while preventing too close shots in terms of duration.
"""

from argparse import ArgumentParser, BooleanOptionalAction
from tensorflow import keras
import cv2
from ultralytics import YOLO
from tqdm import tqdm
import time

from extract_pose import BBoxPoseExtractor, draw_pose, enlarge_bounding_box
from shot_counter import ShotCounter, draw_probs

if __name__ == "__main__":
    parser = ArgumentParser(description="Track and classify shots of multiple people in a video frame by frame.")
    parser.add_argument("video", type=str, help="Path to the input video file.")
    parser.add_argument("-c", "--classifier", type=str, default="padel_fully_connected.keras", help="Path to the shot classification model.")
    parser.add_argument("-y", "--yolo", type=str, default="yolov8n.pt", help="Path to the object detection (YOLO) model.")
    parser.add_argument("-o", "--output", type=str, default="output.mp4", help="Path to save the output video. If not specified, the video will not be saved.")
    parser.add_argument("-s", "--show", action=BooleanOptionalAction, default=False, help="Show the video while processing.")
    parser.add_argument("-v", "--verbose", action=BooleanOptionalAction, default=False, help="Enable verbose output for debugging.")
    args = parser.parse_args()

    yolo = YOLO(args.yolo)
    people = {} # person_id -> (HumanPoseExtractor, ShotCounter)
    classifier = keras.models.load_model(args.classifier)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    if not args.show:
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing", unit="frame")

    frame_num = 0
    MAX_MISSED_FRAMES = fps*2  # Maximum number of frames a person can be missed before being removed from tracking

    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect players using YOLO
        results = yolo.track(frame, persist=True, verbose=args.verbose)[0]

        for i, box in enumerate(results.boxes):
            if box.cls.tolist()[0] != 0:  # Only consider person class
                continue
            
            person_id = box.id
            if person_id is None:
                person_id = i
            else:
                person_id = int(person_id.tolist()[0])  # Convert to integer ID
            
            bbox = box.xyxy.tolist()[0]
            bbox = [int(coord) for coord in bbox]  # Convert to integer coordinates
        
            if person_id not in people:
                # Initialize a new tracker
                people[person_id] = {
                    "pose_extractor": BBoxPoseExtractor(args.verbose),
                    "shot_counter": ShotCounter(fps),
                    "last_seen_frame": frame_num
                }
            else:
                people[person_id]["last_seen_frame"] = frame_num

            # Enlarge the bounding box to ensure the pose is fully contained within the subframe
            bbox = enlarge_bounding_box(bbox)   
            pose = people[person_id]["pose_extractor"].extract(frame, bbox)
            
            if pose is None: # Pose is None if the bounding box is invalid
                continue
            
            pose = pose.reshape(17,3)
            pose_pixels = people[person_id]["pose_extractor"].transform_to_frame_coordinates(pose, bbox)
            
            pose = pose[pose[:, 2] > 0][:,0:2].reshape(1,13*2) # Discard cancelled keypoints, flatten the array before feeding it to the model
            probs = classifier.__call__(pose)[0]

            people[person_id]["shot_counter"].update(probs, frame_num)

            # Display probabilities, shot counter, pose and bounding box
            draw_probs(frame, people[person_id]["shot_counter"].probs.mean(axis=0), bbox)
            people[person_id]["shot_counter"].display(frame, bbox)
            draw_pose(frame, pose_pixels)

            # x1, y1, x2, y2 = bbox
            # big_x1 = int(1.15*x1 - 0.15*x2)
            # big_x2 = int(1.15*x2 - 0.15*x1)
            # big_y1 = int(1.15*y1 - 0.15*y2)
            # big_y2 = int(1.15*y2 - 0.15*y1)

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.rectangle(frame, (big_x1, big_y1), (big_x2, big_y2), (0, 0, 255), 2)

        # Cleanup trackers: remove people that are not detected for MAX_MISSED_FRAMES
        # if frame_num % MAX_MISSED_FRAMES == 0:  # Do that only every 2 seconds
        # inactive_ids = [id for id, obj in people.items() if frame_num - obj["last_seen_frame"] > MAX_MISSED_FRAMES]
        # TODO: Only cleanup empty trackers, otherwise I'll delete ID of people that were detected
        # for person_id in inactive_ids:
        #     if args.verbose:
        #         print(f"Removing person {person_id} after {frame_num - people[person_id]['last_seen_frame']} frames of inactivity.")
        #     del people[person_id]

        frame_num += 1

        # Write the frame to the output video
        if args.output is not None:
            out.write(frame)
        
        if args.show:
            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if k == ord('q') or k == 27:  # ESC or 'q' to quit
                if args.verbose:
                    print("Exiting...")
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

    end = time.time()

    print("Processing complete.")
    print(f"Processed {frame_num} frames in {end - start:.2f} seconds.")
    for person_id, obj in people.items():
        pose_extractor = obj["pose_extractor"]
        shot_counter = obj["shot_counter"]
        print(f"Person {person_id} shot counts: {shot_counter.results}")