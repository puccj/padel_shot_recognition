"""
This script will produce shot annotation on a padel video.
It will output a csv file containing frame id, shot name and player who hit.
If there are two players per field, use arrow keys for the left player and WASD keys for the right player. 
If there is only one player, specify the --single_player flag and use either arrow keys or WASD keys to mark the shots.

In particular:

RIGHT_ARROW_KEY to change the side of shot to right
 LEFT_ARROW_KEY to change the side of shot to left
   UP_ARROW_KEY to select that only one player is present (1vs1 match)

D to mark a shot as FOREHAND
A to mark a shot as BACKHAND
Q to mark a shot as FLAT SMASH
W to mark a shot as 3x SMASH (topspin)
E to mark a shot as RETURN SMASH
S to mark a shot as LOB

C to mark a shot as FOREHAND WALL EXIT (salida de pared)
Z to mark a shot as BACKHAND WALL EXIT (salida de pared)
X to mark a shot as BAJADA (from wall exit)
Left_Alt_KEY to mark a shot as WALL LOB

L to mark a shot as FOREHAND CONTRAPARED
K to mark a shot as BACKHAND CONTRAPARED
P to mark a shot as FOREHAND VOLLEY
O to mark a shot as BANDEJA
I to mark a shot as VIBORA
U to mark a shot as BACKHAND VOLLEY

Y to mark a shot as DROP SHOT
H to mark a shot as RULLO TO THE MESH


SPACE to PAUSE the video
M to JUMP 10 seconds FORWARD
N to JUMP 10 seconds BACKWARD
B to INCREASE playback SPEED
V to DECREASE playback SPEED
ESC  to QUIT the annotation

DELETE to REMOVE the last annotated shot

It is better to hit the key when the player hits the ball.
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2

LEFT_ARROW_KEY = 81
UP_ARROW_KEY = 82
RIGHT_ARROW_KEY = 83
DELETE_KEY = 255
LEFT_ALT_KEY = 130

if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate a video and write a csv file containing padel shots")
    parser.add_argument("video")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speed of the video playback (default: 1.0)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if not cap.isOpened():
        raise IOError("Error opening video stream or file")

    df = pd.DataFrame(columns=["Shot", "FrameId"])

    FRAME_ID = 0
    shot_list = []
    speed = args.speed
    side = "right"

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Frame ID: {FRAME_ID}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(int(1000 / (fps * speed)))

        if k == LEFT_ARROW_KEY:
            side = "left"
            print("Switched to left player")
        elif k == RIGHT_ARROW_KEY:
            side = "right"
            print("Switched to right player")
        elif k == UP_ARROW_KEY:
            side = "one"
            print("Single player mode")

        elif k == ord("d"):
            shot_list.append({"Shot": "forehand", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player forehand")
        elif k == ord("a"):
            shot_list.append({"Shot": "backhand", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player backhand")
        elif k == ord("q"):
            shot_list.append({"Shot": "flat_smash", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player flat smash")
        elif k == ord("w"):
            shot_list.append({"Shot": "topspin_smash", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player topspin smash")
        elif k == ord("e"):
            shot_list.append({"Shot": "return_smash", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player return smash")
        elif k == ord("s"):
            shot_list.append({"Shot": "lob", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player lob")
        elif k == ord("c"):
            shot_list.append({"Shot": "forehand_wall_exit", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player forehand wall exit")
        elif k == ord("z"):
            shot_list.append({"Shot": "backhand_wall_exit", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player backhand wall exit")
        elif k == ord("x"):
            shot_list.append({"Shot": "bajada", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player bajada")
        elif k == LEFT_ALT_KEY:
            shot_list.append({"Shot": "wall_lob", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player wall lob")
        elif k == ord("l"):
            shot_list.append({"Shot": "forehand_contrapared", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player forehand contrapared")
        elif k == ord("k"):
            shot_list.append({"Shot": "backhand_contrapared", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player backhand contrapared")
        elif k == ord("p"):
            shot_list.append({"Shot": "forehand_volley", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player forehand volley")
        elif k == ord("o"):
            shot_list.append({"Shot": "bandeja", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player bandeja")
        elif k == ord("i"):
            shot_list.append({"Shot": "vibora", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player vibora")
        elif k == ord("u"):
            shot_list.append({"Shot": "backhand_volley", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player backhand volley")
        elif k == ord("y"):
            shot_list.append({"Shot": "drop_shot", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player drop shot")
        elif k == ord("h"):
            shot_list.append({"Shot": "rullo_to_mesh", "FrameId": FRAME_ID, "Player": side})
            df = pd.DataFrame.from_records(shot_list)
            print(f"{side.capitalize()} player rullo to the mesh")
        
        elif k == ord(" "):  # Space to pause
            cv2.putText(frame, "Paused. Press any key to continue...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
        elif k == ord("m"):  # M to jump 10 seconds forward
            FRAME_ID += int(fps * 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
            print("Jumped 10 seconds forward")
        elif k == ord("n"):  # N to jump 10 seconds backward
            FRAME_ID -= int(fps * 10)
            if FRAME_ID < 0:
                FRAME_ID = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
            print("Jumped 10 seconds backward")
        elif k == ord("b"):  # B to increase speed
            speed += 0.1
            print(f"Speed increased to {speed}x")
        elif k == ord("v"):  # V to decrease speed
            speed = max(0.1, speed - 0.1)
            print(f"Speed decreased to {speed}x")

        elif k == 27:  # ESC or 'q' to quit
            break

        FRAME_ID += 1

    out_file = f"annotation_{Path(args.video).stem}.csv"
    df.to_csv(out_file, index=False)
    print(f"Annotation file was written to {out_file}")
