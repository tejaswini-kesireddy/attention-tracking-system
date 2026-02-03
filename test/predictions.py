import csv
import time
import os

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

class GazePrediction:
    def __init__(self):
        """
        Initiates the class by opening webcam for video capture, initializing mediapipe facemesh, setting up csv files
        for different gaze directions and displays key information.
        """
        self.mp_face_mesh = mp.solutions.face_mesh  # facemesh using mediapipe
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.draw = mp.solutions.drawing_utils
        self.cam = None
        self.Att_Model = None
        self.prediction_log = []
        self.model_file = os.path.join(MODEL_DIR, "class_monitor_3class_1_new.h5")
        self.log_file = os.path.join(BASE_DIR, "predictions_log.csv")

    def load_model(self):
        """
        Loads the trained model
        """
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model not found: {self.model_file}")
        self.Att_Model = load_model(self.model_file)
        print("Model loaded successfully!")

    def capture(self):
        """
        Detects and captures facial landmarks per each frame and saves them to their respective csv files based on
        the labels.
        """
        if self.Att_Model is None:
            raise RuntimeError("Error: Model not loaded.")
        self.cam = cv2.VideoCapture(0)
        print("Starting prediction... Press ESC or 'q' to stop the prediction")
        start_time = time.time()

        while self.cam.isOpened():
            success, frame = self.cam.read()
            if not success:
                break

            current_time = time.time()
            timestamp_sec = current_time - start_time

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    self.draw.draw_landmarks(frame, face_landmarks)

                    row = []
                    for lm in face_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])

                    if len(row) == 1434:
                        features = np.array(row).reshape(-1, 1434, 1)
                        prediction = self.Att_Model.predict(features, verbose=0)
                        predicted_label = int(np.argmax(prediction, axis=1)[0])
                        print(f"[{predicted_label}]")
                        self.prediction_log.append((round(timestamp_sec, 2), predicted_label))

            cv2.imshow("Real-time Gaze Prediction", frame)

            key = cv2.waitKey(1) & 0xFF
            if self.press_key(key):
                break
        self.calculate_time(start_time)
        self.save_csv()
        self.clean()

    def press_key(self, key):
        """
        Handles given keyboard input to set a label to it or exit the program.

        Args: key (int): The ASCII code of the pressed key.
        """
        if key == 27 or key == ord('q'):
            print("Stopping prediction...")
            return True
        return False

    def calculate_time(self, start_time):
        """
        Calculates the total time of the prediction session in seconds.

        Args: start_time(float): timestamp at the beginning of the session.
        """
        end_time = time.time()
        total = end_time - start_time
        print(f"\nTotal time taken: {total:.3f} seconds")

    def save_csv(self):
        """
        Saves the prediction log to a CSV file.
        """
        if not self.prediction_log:
            print("No predictions to save.")
            return

        with open(self.log_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time(seconds)", "label"])
            writer.writerows(self.prediction_log)
        print("Predictions saved to predictions_log.csv.")

    def clean(self):
        """
        Releases camera and closes cam window.
        """
        if self.cam is not None:
            self.cam.release()
        cv2.destroyAllWindows()


def main():
    """Main function"""
    pred = GazePrediction()
    pred.load_model()
    pred.capture()


if __name__ == "__main__":
    main()
