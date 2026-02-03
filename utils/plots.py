import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PlotGraph:
    def __init__(self, csv_file=None, output_file="histogram.png"):
        """
        Initializes the PlotGraph class with the input CSV file and output graph.

        Args:
        csv_file (str): Path to the CSV file containing prediction logs.
        output_file (str): Name of the output file to save the histogram.
        """
        if csv_file is None:
            csv_file = os.path.join(BASE_DIR, "predictions_log.csv")
        self.csv_file = csv_file
        self.output_file = os.path.join(BASE_DIR, output_file)
        self.label_names = ["Straight", "Left", "Right", "Down", "Up"]

    def prediction_time(self):
        """
        Calculates the total duration of each label.

        Returns: dict: A dictionary of each label (int) with it's the total duration (float) in seconds.
        """
        duration = defaultdict(float)

        with open(self.csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            previous_time = None
            previous_label = None

            for row in reader:
                time = float(row['time(seconds)'])
                label = int(row['label'])
                if previous_label == label:
                    duration[previous_label] += time - previous_time
                else:
                    duration[label] += 0

                previous_time = time
                previous_label = label

        return duration

    def plot_histogram(self):
        """
        Generates and saves a histogram with the total time spent looking in each gaze direction.
        """
        label_duration = self.prediction_time()
        y_axis = [label_duration[i] for i in range(5)]

        plt.figure(figsize=(10, 6))
        plt.bar(self.label_names, y_axis, color='maroon')
        plt.xlabel("Gaze Labels")
        plt.ylabel("Total Time (sec)")
        plt.title("Total Time VS Gaze Labels During Prediction")
        plt.tight_layout()
        plt.savefig(self.output_file)
        print("Success: Graph Saved!")

def main():
    """Main function"""
    plot = PlotGraph()
    plot.plot_histogram()

if __name__ == "__main__":
    main()

