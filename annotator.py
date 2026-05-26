"""
This script will produce shot annotation on a padel video.
It will output a csv file containing frame id, shot name, player who hit and whether the shot was "good" or "bad".
If there are two players per field, use arrow keys for the left player and WASD keys for the right player. 
If there is only one player, specify the --single_player flag and use either arrow keys or WASD keys to mark the shots.

In particular:

RIGHT_ARROW_KEY to change the side of shot to right
 LEFT_ARROW_KEY to change the side of shot to left
   UP_ARROW_KEY to chenge the location of the shot to top       (used if two players are in the same side)
 DOWN_ARROW_KEY to change the location of the shot to bottom    (used if two players are in the same side)

ENTER_KEY to mark a shot as SERVE
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

Y to mark a shot as DROP SHOT (dormillona)
H to mark a shot as RULLO TO THE MESH

CAPS_LOCK to TOGGLE between "good" or "bad" shot
SHIFT to TOGGLE for the last shot

SPACE to PAUSE the video
M to JUMP 10 seconds FORWARD
N to JUMP 5 seconds BACKWARD
. to JUMP 1 frame FORWARD  (while PAUSED)
, to JUMP 1 frame BACKWARD (while PAUSED)
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
DOWN_ARROW_KEY = 84
DELETE_KEY = 255
LEFT_ALT_KEY = 233
ENTER_KEY = 13
CAPS_LOCK_KEY = 229
LEFT_SHIFT_KEY = 225

SHOT_KEYS = {
    ENTER_KEY: "serve",
    ord("d"): "forehand",
    ord("a"): "backhand",
    ord("q"): "flat_smash",
    ord("w"): "topspin_smash",
    ord("e"): "return_smash",
    ord("s"): "lob",
    ord("c"): "forehand_wall_exit",
    ord("z"): "backhand_wall_exit",
    ord("x"): "bajada",
    LEFT_ALT_KEY: "wall_lob",
    ord("l"): "forehand_contrapared",
    ord("k"): "backhand_contrapared",
    ord("p"): "forehand_volley",
    ord("o"): "bandeja",
    ord("i"): "vibora",
    ord("u"): "backhand_volley",
    ord("y"): "drop_shot",
    ord("h"): "rullo_to_mesh"
}

SHOT_COLUMNS = ["Shot", "FrameId", "Player", "Good"]


if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate a video and write a csv file containing padel shots")
    parser.add_argument("video")
    parser.add_argument("-s", "--start", type=int, default=0, help="Start frame")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if not cap.isOpened():
        raise IOError("Error opening video stream or file")

    df = pd.DataFrame(columns=SHOT_COLUMNS)

    FRAME_ID = args.start
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
    shot_list = []
    speed = args.speed
    side = "right"
    good = True

    # Read until video is completed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Frame ID: {FRAME_ID}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {speed:.1f}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Current side: {side}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Shot outcome: {'good' if good else 'bad'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(int(1000 / (fps * speed)))

        if k == LEFT_ARROW_KEY:
            side = "left"
        elif k == RIGHT_ARROW_KEY:
            side = "right"
        elif k == UP_ARROW_KEY:
            side = "top"
        elif k == DOWN_ARROW_KEY:
            side = "bottom"
        elif k == CAPS_LOCK_KEY:
            good = not good

        elif k in SHOT_KEYS:
            shot_name = SHOT_KEYS[k]
            shot_list.append({"Shot": shot_name, "FrameId": FRAME_ID, "Player": side, "Good": good})
            df = pd.DataFrame.from_records(shot_list)
            status = "good" if good else "bad"
            print(f"[FRAME {FRAME_ID}] {side.capitalize()} player {shot_name} ({status})")
        
        elif k == ord(" "):  # Space to pause
            cv2.putText(frame, "Paused. Press Space to continue...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            pause_key = cv2.waitKey(0)
            while pause_key != ord(" "):
                if pause_key == ord("."):  # . to jump 1 frame forward
                    pass
                elif pause_key == ord(","):  # , to jump 1 frame backward
                    FRAME_ID -= 2
                    if FRAME_ID < 0:
                        FRAME_ID = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
                elif pause_key == LEFT_ARROW_KEY:
                    side = "left"
                elif pause_key == RIGHT_ARROW_KEY:
                    side = "right"
                elif pause_key == UP_ARROW_KEY:
                    side = "top"
                elif pause_key == DOWN_ARROW_KEY:
                    side = "bottom"
                elif pause_key == CAPS_LOCK_KEY:
                    good = not good
                elif pause_key == DELETE_KEY:  # DELETE to remove last annotation
                    if shot_list:
                        removed_shot = shot_list.pop()
                        df = pd.DataFrame.from_records(shot_list)
                        print(f"Removed last annotation: {removed_shot}")
                    else:
                        print("No annotations to remove.")
                elif pause_key == LEFT_SHIFT_KEY:  # SHIFT to toggle for the last shot
                    if shot_list:
                        shot_list[-1]["Good"] = not shot_list[-1]["Good"]
                        df = pd.DataFrame.from_records(shot_list)
                        print(f"Toggled last annotation: {shot_list[-1]}")
                    else:
                        print("No annotations to toggle.")
                elif pause_key != -1 and pause_key != ord(" "):
                    print(f"Unrecognized key: {pause_key}")
                
                ret, frame = cap.read()
                FRAME_ID += 1
                cv2.putText(frame, f"Frame ID: {FRAME_ID}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Paused. Press Space to continue...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Speed: {speed:.1f}x", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Current side: {side}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Shot outcome: {'good' if good else 'bad'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Frame", frame)
                pause_key = cv2.waitKey(0)

        elif k == ord("m"):  # M to jump 10 seconds forward
            FRAME_ID += int(fps * 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
        elif k == ord("n"):  # N to jump 5 seconds backward
            FRAME_ID -= int(fps * 5)
            if FRAME_ID < 0:
                FRAME_ID = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
        elif k == ord("b"):  # B to increase speed
            speed += 0.1
        elif k == ord("v"):  # V to decrease speed
            speed = max(0.1, speed - 0.1)

        elif k == 27:  # ESC to quit
            break

        elif k == DELETE_KEY:  # DELETE to remove last annotation
            if shot_list:
                removed_shot = shot_list.pop()
                df = pd.DataFrame.from_records(shot_list)
                print(f"Removed last annotation: {removed_shot}")
            else:
                print("No annotations to remove.")
        elif k == LEFT_SHIFT_KEY:  # SHIFT to toggle for the last shot
            if shot_list:
                shot_list[-1]["Good"] = not shot_list[-1]["Good"]
                df = pd.DataFrame.from_records(shot_list)
                print(f"Toggled last annotation: {shot_list[-1]}")
            else:
                print("No annotations to toggle.")

        elif k != -1:
            print(f"Unrecognized key: {k}")

        FRAME_ID += 1

    out_file = f"annotation_{Path(args.video).stem}.csv"
    df.to_csv(out_file, index=False)
    print(f"Annotation file was written to {out_file}")
