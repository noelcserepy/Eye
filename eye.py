import cv2
import math
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
        self.tracking_face = False
        self.clock = 0
        self.reset_timer = 50
        self.face_dist = 0
        self.depth_multiplier = 1000

        # Eye parameters
        self.eye_radius = 150
        self.iris_radius = 80
        self.pupil_radius = 30
        self.eye_pos = [0, 0]
        self.iris_color = (239, 227, 142)
        self.eye_speed = 25
        self.min_distance = 3
        self.face_pos = [0, 0]
        self.end = 5

        # Eye mask
        self.mask_anim = self.load_mask()
        self.mask = self.mask_anim[0]
        self.line_anim = self.load_line()
        self.anim_frame = 0

        self.run()

    def load_line(self):
        mask_anim = []
        for i in range(13):
            img = cv2.imread(f"./Eye_Anim/Line/Lazy-{i + 1}.png", cv2.IMREAD_GRAYSCALE)
            cv2.resize(img, (self.screen_w, self.screen_h))
            mask_anim.append(img)
        return mask_anim

    def load_mask(self):
        mask_anim = []
        for i in range(13):
            img = cv2.imread(
                f"./Eye_Anim/Mask/Lazy_Mask-{i + 1}.png", cv2.IMREAD_GRAYSCALE
            )
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
            self.tracking_face = True
            self.face_pos = [focus_x, focus_y]
        # If face is not detected: stay put
        else:
            self.tracking_face = False

    def move_eye(self, img, dest):
        # Moves iris smoothly. Speed determined by eye_speed
        dest = self.get_max_dest(dest)
        diff = [
            int(dest[0]) - int(self.eye_pos[0]),
            int(dest[1]) - int(self.eye_pos[1]),
        ]
        distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

        if distance > self.eye_speed:
            x = np.uint8(self.eye_pos[0] + int(diff[0] / distance * self.eye_speed))
            y = np.uint8(self.eye_pos[1] + int(diff[1] / distance * self.eye_speed))
            self.draw_iris(img, x, y)
        if distance < self.min_distance:
            self.draw_iris(img, self.eye_pos[0], self.eye_pos[1])
        else:
            self.draw_iris(img, dest[0], dest[1])

    def get_max_dest(self, dest):
        # Prevents iris from moving outside of the eye mask
        center_diff = [dest[0] - self.center[0], dest[1] - self.center[1]]
        center_dist = math.sqrt(center_diff[0] ** 2 + center_diff[1] ** 2)

        if center_dist > self.eye_radius:
            max_x = self.center[0] + int(center_diff[0] / center_dist * self.eye_radius)
            max_y = self.center[1] + int(center_diff[1] / center_dist * self.eye_radius)
            return [max_x, max_y]
        else:
            return dest

    def draw_iris(self, img, x, y):
        cv2.circle(img, (x, y), self.iris_radius, (255, 255, 255), 3)
        cv2.circle(img, (x, y), self.pupil_radius, (255, 255, 255), cv2.FILLED)
        self.eye_pos = [x, y]

    def downsample(self, img):
        img = cv2.resize(img, (32, 32))
        return img

    def animate_eye(self):
        self.mask = self.mask_anim[self.anim_frame]
        self.anim_frame += 1
        if self.anim_frame >= self.end:
            self.anim_frame = self.end

    def return_eye(self):
        if self.tracking_face:
            self.clock = 0
        if not self.tracking_face:
            self.clock += 1
        if self.clock > self.reset_timer:
            self.face_pos = self.center

    def create_depth(self):
        if not self.face_dist:
            return
        center_diff = [
            self.face_pos[0] - self.center[0],
            self.face_pos[1] - self.center[1],
        ]

        face_dist_in_pixels = 1 / self.face_dist * self.depth_multiplier
        xy_multiplier = self.eye_radius // face_dist_in_pixels

        x = int(center_diff[0] * xy_multiplier)
        y = int(center_diff[1] * xy_multiplier)
        # x = int(self.center[0]) + int(
        #     center_diff[0] * math.sqrt(self.face_dist) / self.depth_multiplier
        # )
        # y = int(self.center[1]) + int(
        #     center_diff[1] * math.sqrt(self.face_dist) / self.depth_multiplier
        # )
        self.face_pos = [x, y]

    def run(self):
        while True:
            _, img = self.cap.read()
            img = cv2.flip(img, 1)
            self.find_face(img)
            self.create_depth()
            self.return_eye()

            # draw img
            img = np.zeros((self.screen_h, self.screen_w), np.uint8)

            self.move_eye(img, self.face_pos)
            self.animate_eye()

            # Mask
            img = cv2.bitwise_and(img, img, mask=self.mask)
            img = cv2.addWeighted(img, 1, self.line_anim[self.anim_frame], 1, 0)

            cv2.imshow("Output", img)
            cv2.waitKey(1)


if __name__ == "__main__":
    Eye(cv2.VideoCapture(0))
