import cv2
import math
import random
from datetime import date, datetime
import numpy as np


class Eye:
    def __init__(self, cap):
        # Video Capture and Frame
        self.cap = cap
        self.screen_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.screen_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Width: {self.screen_w} Height: {self.screen_h}")
        self.center = [self.screen_w // 2, self.screen_h // 2]
        self.canvas = np.zeros((self.screen_h, self.screen_w), np.uint8)

        # Tracking
        self.is_tracking_face = False
        self.clock = 0
        self.reset_timer = 50
        self.face_dist = 0
        self.last_tracking_time = None

        # Eye parameters
        self.eye_radius = 150
        self.iris_radius = 80
        self.pupil_radius = 30
        self.eye_pos = [self.center[0], self.center[1]]
        self.iris_color = (239, 227, 142)
        self.eye_speed = 20
        self.min_distance = 3
        self.face_pos = [0, 0]

        self.min_y_pos = self.center[1] - self.eye_radius
        self.max_y_pos = self.center[1] + self.eye_radius

        self.angle = 0

        # Animation
        self.ul_line = self.load_anims("./Anims/Upper_Lid_Line/Upper_Lid_Line-")
        self.ul_mask = self.load_anims("./Anims/Upper_Lid_Mask/Upper_Lid_Mask-")
        self.ll_line = self.load_anims("./Anims/Lower_Lid_Line/Lower_Lid_Line-")
        self.ll_mask = self.load_anims("./Anims/Lower_Lid_Mask/Lower_Lid_Mask-")
        self.mask = self.ul_mask[0]
        self.ul_frame = 0
        self.ll_frame = 0
        self.anim_end_frame = 16

        self.expressions = {
            "stare": [0, 0],
            "open": [2, 2],
            "squint": [6, 6],
            "amused": [1, 6],
            "lazy": [6, 2],
            "pissed": [8, 0],
            "closed": [15, 0],
        }
        self.current_expression = [0, 0]

        self.run()

    def load_anims(self, path):
        mask_anim = []
        for i in range(16):
            img = cv2.imread(f"{path}{i + 1}.png", cv2.IMREAD_GRAYSCALE)
            cv2.resize(img, (self.screen_w, self.screen_h))
            mask_anim.append(img)
        return mask_anim

    def find_face(self, img):
        # Face detection algo
        face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, 1.2, 8)

        # Extract face coordinates
        face_found = False
        for (x, y, w, h) in faces:
            focus_x = x + w // 2
            focus_y = y + h // 3
            self.face_dist = math.sqrt(w * h)
            face_found = True

        # If face is detected: track face
        if face_found:
            self.is_tracking_face = True
            self.face_pos = [focus_x, focus_y]
        # If face is not detected: stay put
        else:
            self.is_tracking_face = False

    def move_eye(self, img, destination):
        # Moves iris smoothly. Speed determined by eye_speed
        self.center_diff = [
            destination[0] - self.center[0],
            destination[1] - self.center[1],
        ]
        self.center_dist = math.sqrt(
            self.center_diff[0] ** 2 + self.center_diff[1] ** 2
        )

        # Get angle used for forming Iris ellipse
        self.get_angle()

        destination = self.get_max_dest(destination)

        diff = [
            int(destination[0]) - int(self.eye_pos[0]),
            int(destination[1]) - int(self.eye_pos[1]),
        ]
        distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

        if distance > self.eye_speed:
            x = np.uint8(self.eye_pos[0] + int(diff[0] / distance * self.eye_speed))
            y = np.uint8(self.eye_pos[1] + int(diff[1] / distance * self.eye_speed))
            self.draw_iris(img, x, y)
        if distance < self.min_distance:
            self.draw_iris(img, self.eye_pos[0], self.eye_pos[1])
        else:
            self.draw_iris(img, destination[0], destination[1])

    def get_max_dest(self, destination):
        # Prevents iris from moving outside of the eye mask
        if self.center_dist > self.eye_radius:
            max_x = self.center[0] + int(
                self.center_diff[0] / self.center_dist * self.eye_radius
            )
            max_y = self.center[1] + int(
                self.center_diff[1] / self.center_dist * self.eye_radius
            )
            return [max_x, max_y]
        else:
            return destination

    def get_angle(self):
        # Calculates the angle of the eye in degrees
        if self.center_dist != 0:
            o_by_h = self.center_diff[1] / self.center_dist
            angle = math.degrees(math.asin(o_by_h))

            # Calculating so that increasing degrees == moving around the circle
            # 0-90
            if self.center_diff[0] >= 0 and self.center_diff[1] <= 0:
                angle = angle + 90
            # 90-180
            if self.center_diff[0] >= 0 and self.center_diff[1] >= 0:
                angle = angle + 90
            # 180-270
            if self.center_diff[0] <= 0 and self.center_diff[1] >= 0:
                angle = 270 - angle
            # 270-360
            if self.center_diff[0] <= 0 and self.center_diff[1] <= 0:
                angle = 270 - angle

            self.angle = angle

    def draw_iris(self, img, x, y):
        squash_factor = 1 - (self.center_dist / 250)
        squash_factor = max(0.3, min(squash_factor, 1))

        cv2.ellipse(
            img,
            (x, y),
            (self.iris_radius, int(self.iris_radius * squash_factor)),
            self.angle,
            0,
            360,
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.ellipse(
            img,
            (x, y),
            (self.pupil_radius, int(self.pupil_radius * squash_factor)),
            self.angle,
            0,
            360,
            (0, 0, 0),
            cv2.FILLED,
        )
        # cv2.circle(img, (x, y), self.iris_radius, (255, 255, 255), cv2.FILLED)
        # cv2.circle(img, (x, y), self.pupil_radius, (0, 0, 0), cv2.FILLED)
        self.eye_pos = [x, y]

    def downsample(self, img):
        img = cv2.resize(img, (32, 32))
        return img

    def sync_lid_with_eye(self):
        eye_height = self.eye_pos[1] - self.min_y_pos
        total_y_travel = self.max_y_pos - self.min_y_pos
        portion_of_eye_travelled = eye_height / (total_y_travel)
        upper_lid_offset = 4
        lower_lid_offset = 2

        # Relating lid animation frame to eye position and offsetting it to make it look good
        lid_frame_upper = round(portion_of_eye_travelled * 16)
        lid_frame_upper -= upper_lid_offset
        lid_frame_upper = max(0, min(lid_frame_upper, 15))
        self.current_expression[0] = lid_frame_upper

        lid_frame_lower = round(portion_of_eye_travelled * 8)
        lid_frame_lower = 8 - lid_frame_lower - lower_lid_offset
        lid_frame_lower = max(0, min(lid_frame_lower, 8))
        self.current_expression[1] = lid_frame_lower

    def animate_eye(self):
        self.sync_lid_with_eye()

        if self.ul_frame > self.current_expression[0]:
            self.ul_frame -= 1
        if self.ul_frame < self.current_expression[0]:
            self.ul_frame += 1
        if self.ll_frame > self.current_expression[1]:
            self.ll_frame -= 1
        if self.ll_frame < self.current_expression[1]:
            self.ll_frame += 1

        self.mask = cv2.bitwise_and(
            self.ul_mask[self.ul_frame], self.ll_mask[self.ll_frame]
        )

    def reset_eye_position(self):
        # Returns eye back to center position after not detecting face for some time
        if self.is_tracking_face:
            return
        if not self.last_tracking_time:
            self.last_tracking_time = datetime.now()

        elapsed = (datetime.now() - self.last_tracking_time).total_seconds()
        if elapsed >= 2:
            self.face_pos = self.center
            self.last_tracking_time = None
            elapsed = 0

    def create_depth(self):
        # Makes the eye look at you more realistically
        if not self.is_tracking_face:
            return

        center_diff = [
            self.face_pos[0] - self.center[0],
            self.face_pos[1] - self.center[1],
        ]
        x = int(self.center[0] + center_diff[0] * 0.6)
        y = int(self.center[1] + center_diff[1] * 0.6)

        self.face_pos = [x, y]

    def lid_control(self):
        key = cv2.waitKey(1)

        # Upper
        if key == ord("h"):
            self.ul_frame += 1
        if key == ord("j"):
            self.ul_frame -= 1
        if 0 > self.ul_frame:
            self.ul_frame = 0
        if self.ul_frame > 15:
            self.ul_frame = 15

        # Lower
        if key == ord("k"):
            self.ll_frame += 1
        if key == ord("l"):
            self.ll_frame -= 1
        if 0 > self.ll_frame:
            self.ll_frame = 0
        if self.ll_frame > 15:
            self.ll_frame = 15

        # Change expression
        if key == ord("e"):
            expr_list = [*self.expressions.values()]
            self.current_expression = random.choice(expr_list)

    def run(self):
        while True:
            _, img = self.cap.read()
            img = cv2.flip(img, 1)
            self.find_face(img)

            self.create_depth()

            self.reset_eye_position()

            self.animate_eye()

            # Draw img
            img = np.zeros((self.screen_h, self.screen_w), np.uint8)

            # Draw eye ring
            cv2.circle(
                img,
                (self.center[0], self.center[1]),
                self.eye_radius,
                (255, 255, 255),
                4,
            )

            # Draw eyeball
            self.move_eye(img, self.face_pos)

            # Mask
            img = cv2.bitwise_and(img, img, mask=self.mask)

            # Add img, lower and upper lid lines together
            img = cv2.addWeighted(img, 1, self.ul_line[self.ul_frame], 1, 0)
            img = cv2.addWeighted(img, 1, self.ll_line[self.ll_frame], 1, 0)

            cv2.imshow("Output", img)

            # Manual controls over eyelid positions
            self.lid_control()


if __name__ == "__main__":
    Eye(cv2.VideoCapture(0))
