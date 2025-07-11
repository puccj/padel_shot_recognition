import numpy as np
import cv2

class ShotCounter:
    """Basic shot counter with a shot history"""

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

    def display(self, frame, bbox=None):
        """ Display shot count, colorize last shot in green.
        If a bbox is provided, the text is displayed below the bbox, otherwise, it is displayed at the bottom of the frame.
        """

        if bbox is None:
            spacing = 40
            x_min = 20
            y_bottom = frame.shape[0]
            size = 1
        else:
            spacing = 20
            x_min = bbox[0] + 20
            y_bottom = bbox[3] + 20 + spacing*3
            size = 0.5

        cv2.putText(
            frame,
            f"Backhands = {self.nb_backhands}",
            (x_min, y_bottom - 20 - spacing*2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            color= (0,255,0) if (self.last_shot == "backhand" and self.frames_since_last_shot < 30) else (0,0,255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Forehands = {self.nb_forehands}",
            (x_min, y_bottom - 20 - spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            color= (0,255,0) if (self.last_shot == "forehand" and self.frames_since_last_shot < 30) else (0,0,255),
            thickness=2,
        )
        cv2.putText(
            frame,
            f"Smashes = {self.nb_smashes}",
            (x_min, y_bottom - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=size,
            color= (0,255,0) if (self.last_shot == "smash" and self.frames_since_last_shot < 30) else (0,0,255),
            thickness=2,
        )

def draw_probs(frame, probs, bbox=None):
    """Draw vertical bars representing probabilities. 
    If a bbox is provided, the bars are drawn near the bbox, otherwise they are drawn at a fixed position."""

    if bbox is None:
        bar_width = 30
        bar_height = 170
        space_between_bars = 54
        margin_above_bar = 30
        bar_x = 1070
        text_x = 1075
        text_y = 230
        text_size = 1
        thickness = 3
    else:
        bar_width = 22
        bar_height = 128
        space_between_bars = 40
        margin_above_bar = bbox[1] - bar_height - 30
        bar_x = bbox[0] + 40
        text_x = bar_x + bar_width // 3
        text_y = bbox[1] - 10
        text_size = 0.5
        thickness = 2

    cv2.putText(frame, "B", (text_x                       , text_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,255), thickness)
    cv2.putText(frame, "F", (text_x + space_between_bars  , text_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,255), thickness)
    cv2.putText(frame, "N", (text_x + space_between_bars*2, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,255), thickness)
    cv2.putText(frame, "S", (text_x + space_between_bars*3, text_y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,255), thickness)
    
    for i, prob in enumerate(probs):
        x_min = bar_x + space_between_bars * i
        y_max = bar_height + margin_above_bar
        
        # Filled rectangle for the probability bar, empty rectangle for the border
        cv2.rectangle(frame, (x_min, int(y_max - bar_height * prob)), (x_min + bar_width, y_max), ( 0 , 0 ,255), -1)
        cv2.rectangle(frame, (x_min, int(margin_above_bar))         , (x_min + bar_width, y_max), (255,255,255), 1)

    return frame