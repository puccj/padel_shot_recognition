"""
This script will produce shot annotation on a padel video.
It will output a csv file containing frame id, shot name and player who hit.
If there are two players per field, use arrow keys for the left player and WASD keys for the right player. 
If there is only one player, specify the --single_player flag and use either arrow keys or WASD keys to mark the shots.

In particular:

RIGHT_ARROW_KEY to mark a shot as FOREHAND
 LEFT_ARROW_KEY to mark a shot as BACKHAND
   UP_ARROW_KEY to mark a shot as SMASH

D_KEY to mark a shot as FOREHAND
A_KEY to mark a shot as BACKHAND
W_KEY to mark a shot as SMASH

It is better to hit the key when the player hits the ball.
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2

LEFT_ARROW_KEY = 81
UP_ARROW_KEY = 82
RIGHT_ARROW_KEY = 83


if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate a video and write a csv file containing padel shots")
    parser.add_argument("video")
    parser.add_argument("--single_player", 
                        action="store_const", 
                        default=False, 
                        help="Set to True if you want to annotate a video with a single player (1vs1 match)")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speed of the video playback (default: 1.0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if not cap.isOpened():
        raise IOError("Error opening video stream or file")

    df = pd.DataFrame(columns=["Shot", "FrameId"])

    FRAME_ID = 0
    single_player = args.single_player
    your_list = []
    speed = args.speed

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Frame ID: {FRAME_ID}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(int(1000 / (fps * speed)))

        if k == ord(" "):  # Space to pause
            cv2.putText(frame, "Paused. Press any key to continue...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)

        if k == RIGHT_ARROW_KEY:
            your_list.append({"Shot": "forehand", "FrameId": FRAME_ID, "Player": "one" if single_player else "right"})
            df = pd.DataFrame.from_records(your_list)
            print("Forehand" if single_player else "Right player forehand")
        elif k == LEFT_ARROW_KEY:
            your_list.append({"Shot": "backhand", "FrameId": FRAME_ID, "Player": "one" if single_player else "right"})
            df = pd.DataFrame.from_records(your_list)
            print("Backhand" if single_player else "Right player backhand")
        elif k == UP_ARROW_KEY:
            your_list.append({"Shot": "smash", "FrameId": FRAME_ID, "Player": "one" if single_player else "right"})
            df = pd.DataFrame.from_records(your_list)
            print("Smash" if single_player else "Right player smash")
        # elif k == DOWN_ARROW_KEY:  # lob
        #     your_list.append({"Shot": "lob", "FrameId": FRAME_ID, "Player": "right" if single_player else "one"})
        #     df = pd.DataFrame.from_records(your_list)
        #     print("Add lob")

        elif k == ord("d"):
            your_list.append({"Shot": "forehand", "FrameId": FRAME_ID, "Player": "one" if single_player else "left"})
            df = pd.DataFrame.from_records(your_list)
            print("Forehand" if single_player else "Left player forehand")
        elif k == ord("a"):
            your_list.append({"Shot": "backhand", "FrameId": FRAME_ID, "Player": "one" if single_player else "left"})
            df = pd.DataFrame.from_records(your_list)
            print("Backhand" if single_player else "Left player backhand")
        elif k == ord("w"):
            your_list.append({"Shot": "smash", "FrameId": FRAME_ID, "Player": "one" if single_player else "left"})
            df = pd.DataFrame.from_records(your_list)
            print("Smash" if single_player else "Left player smash")
        
        # Jump 10 seconds forward and backward
        elif k == ord("k"):
            FRAME_ID += int(fps * 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
            print("Jumped 10 seconds forward")
        elif k == ord("j"):
            FRAME_ID -= int(fps * 10)
            if FRAME_ID < 0:
                FRAME_ID = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
            print("Jumped 10 seconds backward")
        elif k == ord("c"):
            speed += 0.1
            print(f"Speed increased to {speed}x")
        elif k == ord("x"):
            speed = max(0.1, speed - 0.1)
            print(f"Speed decreased to {speed}x")


        if k == 27 or k == ord("q"):  # ESC or 'q' to quit
            break

        FRAME_ID += 1

    out_file = f"annotation_{Path(args.video).stem}.csv"
    df.to_csv(out_file, index=False)
    print(f"Annotation file was written to {out_file}")
