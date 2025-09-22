import numpy as np
import pandas as pd
from statistics import mode
from datetime import datetime, timedelta, timezone
import pytz

gps_epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
leap_seconds = 19
pacific = pytz.timezone("America/Los_Angeles")


class VelocityCalculator:
    def __init__(self, df):
        self.df = df

    def calculate_velocity(self):
        diff = self.df.diff()

        dx = diff["easting"]
        dy = diff["northing"]

        time_diff = diff["GPS_TOW"] / 1000

        vx = np.where(time_diff != 0, dx / time_diff, 0)
        vy = np.where(time_diff != 0, dy / time_diff, 0)

        velocity = np.sqrt(vx**2 + vy**2)

        # Set the first row's velocity to 0
        vx[0], vy[0], velocity[0] = 0, 0, 0
        return velocity, vx, vy


def parse_harvest_date(datestr):
    return datetime.strptime(datestr, "%m-%d-%y").replace(tzinfo=pytz.utc)


# Function to compute datetime from harvest_date and TOW
def compute_datetime(harvest_stamp):
    harvest_date = parse_harvest_date(harvest_stamp["harvest_date"])
    # Snap to midnight UTC to get start of that date
    harvest_midnight = harvest_date.replace(hour=0, minute=0, second=0, microsecond=0)

    # Compute the GPS week number dynamically from the date
    gps_week = int((harvest_midnight - gps_epoch).total_seconds() // 604800)

    # Now compute GPS time (week + tow)
    gps_tow_sec = harvest_stamp["stamps"] / 1000.0
    gps_total_seconds = gps_week * 604800 + gps_tow_sec
    utc_time = gps_epoch + timedelta(seconds=gps_total_seconds - leap_seconds)
    return utc_time.astimezone(pacific)


def calculate_durations(gt_pick_list, cart_data):
    durations = []
    for start_idx, length in gt_pick_list:
        start_time = cart_data["datetime_pacific"].iloc[start_idx]
        end_time = cart_data["datetime_pacific"].iloc[start_idx + length - 1]
        duration = (end_time - start_time).total_seconds() / 60  # Convert to minutes
        durations.append(duration)
    return durations


def replace_largest_breaks_with_median(no_pick_durations, no_breaks):
    def find_largest_indices(values, x):
        return np.argsort(values)[-x:]

    median_no_pick_duration = np.median(no_pick_durations)
    if no_breaks > 0:
        # Replace the largest values with the median
        largest_indices = find_largest_indices(no_pick_durations, no_breaks)
        for index in largest_indices:
            no_pick_durations[index] = median_no_pick_duration
    return no_pick_durations


class CreateDataset:
    @staticmethod
    def compute_activity_segment_mode(activity_labels):
        """
        Compute the mode for segments of activity labels.

        Parameters:
        activity_labels (numpy.ndarray): Array of activity labels.

        Returns:
        numpy.ndarray: Array of modes for each segment.
        """
        # Reshape the array into segments of 100 elements each
        reshaped_segments = activity_labels.reshape(-1, 100)

        # Compute the mode for each segment
        segment_modes = []
        for segment in reshaped_segments:
            mode_value = mode(segment)
            segment_modes.append(mode_value)

        segment_modes_array = np.asarray(segment_modes).reshape(-1, 1)
        return segment_modes_array

    @staticmethod
    def create_dataset(features, labels, time_steps=1, step=1):
        """
        Create a dataset by segmenting features and labels into overlapping windows.

        Parameters:
        features (pandas.DataFrame): The input features.
        labels (pandas.Series): The corresponding labels.
        time_steps (int): The number of time steps in each segment.
        step (int): The step size for the sliding window.

        Returns:
        numpy.ndarray: Segmented feature windows.
        numpy.ndarray: Segmented label windows with computed modes.
        """
        feature_segments, label_segments = [], []
        for start_idx in range(0, len(features) - time_steps, step):
            feature_window = features.iloc[start_idx : (start_idx + time_steps)].values
            label_window = labels.iloc[start_idx : start_idx + time_steps].values
            prepared_labels = CreateDataset.compute_activity_segment_mode(label_window)
            feature_segments.append(feature_window)
            label_segments.append(prepared_labels)
        return np.array(feature_segments), np.array(label_segments)
