import csv
import cv2
import os
import mediapipe as mp


class DataCollection:
    def __init__(self):
        """
        Initiates the class by opening webcam for video capture, initializing mediapipe facemesh, setting up csv files
        for different gaze directions and displays key information.
        """
        self.mp_face_mesh = mp.solutions.face_mesh  # facemesh using mediapipe
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.draw = mp.solutions.drawing_utils

        self.cam = cv2.VideoCapture(0)

        self.labels = ["looking straight", "looking left", "looking right", "looking down", "looking up"]
        self.csv_files = {}
        for each in self.labels:
            filename = f"{each.split()[1]}.csv"
            self.csv_files[each] = open(filename, mode="a", newline="")

        self.csv_writers = {}
        for label, f in self.csv_files.items():
            self.csv_writers[label] = csv.writer(f)

        self.headers_written = {}
        for label, file in self.csv_files.items():
            self.headers_written[label] = os.stat(file.name).st_size > 0
        self.label = None
        print("Press 'l' for left, 'r' for right, 's' for straight, 'u' for up, 'd' for down. Press ESC to stop.")

    def capture(self):
        """
        Detects and captures facial landmarks per each frame and saves them to their respective csv files based on the labels.
        """
        while self.cam.isOpened():
            success, frame = self.cam.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)  # flipping for mirror view
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # converting to rgb
            results = self.face_mesh.process(rgb)

            if self.label:
                cv2.putText(frame, f"Label: {self.label}", (10, 30),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 128), 2)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.draw.draw_landmarks(frame, face_landmarks)

                    # 468 landmarks for face
                    # Get x, y, z values of all 468 landmarks (3*468 = 1404)
                    row = []
                    for each in face_landmarks.landmark:
                        row += [each.x, each.y, each.z]

                    if self.label:
                        writer = self.csv_writers[self.label]
                        if not self.headers_written[self.label]:
                            header = []
                            for i in range(len(row) // 3):
                                header += [f"x_{i}", f"y_{i}", f"z_{i}"]
                            writer.writerow(header)
                            self.headers_written[self.label] = True

                        writer.writerow(row)

            cv2.imshow("Data Collection - Gaze Labels", frame)
            key = cv2.waitKey(1) & 0xFF
            self.press_key(key)

    def press_key(self, key):
        """
        Handles given keyboard input to set a label to it or exit the program.

        Args: key (int): The ASCII code of the pressed key.
        """
        if key == 27:  # ASCII code for ESC to exit
            self.clean()
            exit()
        elif key == ord('s'):
            self.label = "looking straight"
        elif key == ord('l'):
            self.label = "looking left"
        elif key == ord('r'):
            self.label = "looking right"
        elif key == ord('d'):
            self.label = "looking down"
        elif key == ord('u'):
            self.label = "looking up"

    def clean(self):
        """
        Releases camera and closes cam window.
        """
        self.cam.release()
        for f in self.csv_files.values():
            f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    mp_data_collection = DataCollection()
    mp_data_collection.capture()
