import json
import numpy as np
from numpy.lib.arraysetops import intersect1d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression, RANSACRegressor
import time
from shapely.geometry import Polygon, Point
import pwlf
from scipy.signal import medfilt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# cnn related
import tensorflow as tf
from sklearn.preprocessing import RobustScaler


M2FT = 3.28084
TIME_STEPS = 9600
STEP = 9600


########################################################
# Load field skeleton map from a text file
# The data is saved as a python dictionary
# The elements are converted into numpy array
########################################################
def load_map(path):

    data_loaded = json.load(open(path))
    # convert data into numpy array
    data_loaded["odom_NDs"] = np.array(data_loaded["odom_NDs"])
    data_loaded["odom_FDs"] = np.array(data_loaded["odom_FDs"])
    data_loaded["utm_NDs"] = np.array(data_loaded["utm_NDs"])
    data_loaded["utm_FDs"] = np.array(data_loaded["utm_FDs"])
    data_loaded["llh_NDs"] = np.array(data_loaded["llh_NDs"])
    data_loaded["llh_FDs"] = np.array(data_loaded["llh_FDs"])
    data_loaded["odom_T_utm"] = np.array(data_loaded["odom_T_utm"])

    return data_loaded


########################################################
# transform 2D points from an old frame to a new frame
# new_T_old: 4 by 4
# points_old: N by 2
########################################################
def transform_points_to_new_frame(new_T_old, points_old):
    points_old_transfered = points_old.copy()
    points_old_transfered_T = np.concatenate([points_old_transfered, np.ones([len(points_old_transfered), 2])], axis=1).T
    points_old_transfered_T = new_T_old.dot(points_old_transfered_T)
    points_old_transfered = points_old_transfered_T.T
    points_new = points_old_transfered[:, :2]

    return points_new


##########################################################
# Append UTM and field coordinates into the data set
# Output: Raw data with UTM and field coordinates:
#   [`GPS_TOW`, `LAT`, `LON`, `HEIGHT`, `ax`, `ay`, `az`, `raw_mass`, `easting`, `northing`, `x`, `y`]
##########################################################
def augment_field_coords(raw_data, odom_T_utm):
    data_augmented = {}
    for cart_id in raw_data.keys():
        if len(raw_data[cart_id]) > 0:
            utm_points = np.copy(raw_data[cart_id][:, -2:])
            odom_points = transform_points_to_new_frame(odom_T_utm, utm_points)
            data_augmented[cart_id] = np.hstack((raw_data[cart_id], odom_points))

    return data_augmented


# header: "GPS_TOW	LAT	LON	HEIGHT	ax	ay	az	raw_mass easting northing x y"
def create_dataset(X, time_steps=1, step=1):
    Xs, ys = [], []
    for i in range(0, len(X), step):
        v = X[i : (i + time_steps)]
        if len(v) < time_steps:
            v = np.pad(v, ((0, time_steps - len(v)), (0, 0)), mode="constant")
        Xs.append(v)
    return np.array(Xs)


def prepare_test(cart_data):
    train_columns = cart_data[:, 4:8]  # ax	ay	az	raw_mass
    scaler = RobustScaler()
    scaler = scaler.fit(train_columns)
    train_columns = scaler.transform(train_columns)

    X_test = create_dataset(train_columns, TIME_STEPS, STEP)
    return X_test


def run_model(X_test, model_path):
    model = tf.keras.models.load_model(model_path)
    y_pred = model.predict(X_test)
    return y_pred


def get_grouped_class_lengths(cart_data, column_name, target_value):
    """Take the cart data and class and find the groups of index and length of each group

    Args:
        cart_data (dataframe): cart_data
        column_name ("pick_status): _description_
        target_value (class): pick 1 and nopick 0

    Returns:
        _type_: result list, median
    """
    class_arr = np.array(cart_data[column_name])
    grouped = cart_data[column_name].groupby((cart_data[column_name] != cart_data[column_name].shift()).cumsum())

    lengths = []
    indices = []

    # Iterate through each group
    for _, group in grouped:
        if group.iloc[0] == target_value:  # Only consider groups starting with target_value
            count_values = (group == target_value).sum()  # Count occurrences of target_value in the group
            if count_values > 0:
                lengths.append(count_values)
                indices.append(group.index[0])  # Get the index of the first occurrence

    result_list = list(zip(indices, lengths))
    return result_list, np.median(lengths) if lengths else None


########################################################
# Remove the data points outside of the picking rows
# pick_area: shapely polygon
########################################################
def identify_range_outliers(raw_data, pick_area, x_offset=0.0):
    filter_data = {}
    for cart_id in raw_data.keys():
        odom_points = raw_data[cart_id][:, 2:4]
        for i, point in enumerate(odom_points):
            if not pick_area.contains(Point(point)):
                raw_data[cart_id][
                    i, -1
                ] = "NoPick"  # Mark as NoPick in the last column if the point is outside the picking area
                raw_data[cart_id][
                    i, -2
                ] = "NoPick"  # Mark as NoPick in the last column if the point is outside the picking area
        if x_offset != 0:
            raw_data[cart_id][:, -3] -= x_offset  # Adjust x-coordinate if x_offset is provided
        filter_data[cart_id] = raw_data[cart_id]

    return filter_data


def identify_range_outliers_raw(raw_data, pick_area, x_offset=0.0):
    filter_data = {}
    for cart_id in raw_data.keys():
        odom_points = raw_data[cart_id][:, -3:-1]
        for i, point in enumerate(odom_points):
            if not pick_area.contains(Point(point)):
                raw_data[cart_id][i, -1] = 0  # Mark as NoPick in the last column if the point is outside the picking area
        if x_offset != 0:
            raw_data[cart_id][:, -3] -= x_offset  # Adjust x-coordinate if x_offset is provided
        filter_data[cart_id] = raw_data[cart_id]

    return filter_data


def remove_speed_outliers(
    raw_data,
    x_speed_limit=0.25,
    y_speed_limit=0.7,
    lowerlimity=0.0,
    upperlimity=80.0,
    lowerlimitx=-200.0,
    upperlimitx=0.0,
):
    tolerance = 0.4
    x_speeds = {}
    y_speeds = {}
    filter_data = {}
    for cart_id in raw_data.keys():
        # remove x speed outliers
        xs = raw_data[cart_id][:, -2]
        ys = raw_data[cart_id][:, -1]
        ts = raw_data[cart_id][:, 0] / 1000
        x_speed = np.abs(np.diff(xs) / np.diff(ts))
        y_speed = np.abs(np.diff(ys) / np.diff(ts))
        small_x_and_y_speed_idxs = np.where((x_speed < x_speed_limit) & (y_speed < y_speed_limit))[0]

        if len(raw_data[cart_id][small_x_and_y_speed_idxs]) > 0:

            filter_data[cart_id] = raw_data[cart_id][small_x_and_y_speed_idxs]
            xs = filter_data[cart_id][:, -2]
            ys = filter_data[cart_id][:, -1]
            limit_y_and_limit_y_idxs = np.where((ys > (lowerlimity + tolerance)) & (ys < (upperlimity - tolerance)))[0]
            if len(raw_data[cart_id][limit_y_and_limit_y_idxs]) > 0:
                filter_data[cart_id] = filter_data[cart_id][limit_y_and_limit_y_idxs]
                xs = filter_data[cart_id][:, -2]
                ys = filter_data[cart_id][:, -1]
                limit_x_and_limit_x_idxs = np.where((xs > lowerlimitx) & (xs < upperlimitx))[0]
                if len(raw_data[cart_id][limit_x_and_limit_x_idxs]) > 0:
                    filter_data[cart_id] = filter_data[cart_id][limit_x_and_limit_x_idxs]
            # print(type(filter_data[cart_id]))
            x_speeds[cart_id] = x_speed
            y_speeds[cart_id] = y_speed

    return filter_data, x_speeds, y_speeds


########################################################
# Remove the data points outside of the picking rows
# pick_area: shapely polygon
########################################################
def remove_range_outliers(raw_data, pick_area, x_offset=0.0):
    filter_data = {}
    for cart_id in raw_data.keys():
        #         print(cart_id)
        #         print("**************")
        odom_points = raw_data[cart_id][:, -2:]
        in_bound_idxs = []
        for i, point in enumerate(odom_points):
            if pick_area.contains(Point(point)):
                in_bound_idxs.append(i)
        if len(raw_data[cart_id][in_bound_idxs]) > 0:
            filter_data[cart_id] = raw_data[cart_id][in_bound_idxs]  # selecting indices which are included
            filter_data[cart_id][:, 9] = in_bound_idxs
            if x_offset != 0:
                filter_data[cart_id][:, -2] -= x_offset

    return filter_data


##########################################################
# Get pick area based on a field shape given tolerance
# The returned pick_area is in shapely polygon
##########################################################
def get_pick_area(odom_downs, odom_ups, tolerance=1):
    # an hexagon for the map
    up_idxs = [0, 159, 162]
    down_idxs = [162, 100, 99, 98, 97, 3, 2, 0]
    vertices_up = odom_ups[up_idxs]
    vertices_up[:, 1] -= tolerance
    vertices_down = odom_downs[down_idxs]
    vertices_down[:, 1] += tolerance
    vertices = np.vstack((vertices_up, vertices_down))
    print(vertices.shape)
    pick_area = Polygon(vertices)

    return pick_area


def get_pick_area_santamaria_24(odom_downs, odom_ups, tolerance_x=1, tolerance_y=0):
    # an hexagon for the map
    # odom_downs=odomNDs
    # odom_ups=odomFDs
    up_idxs = [
        124,
        123,
        121,
        119,
        113,
        105,
        95,
        85,
        75,
        67,
        61,
        57,
        55,
        51,
        43,
        35,
        25,
        13,
        7,
        5,
        3,
        1,
        0,
    ]
    down_idxs = [
        0,
        5,
        6,
        7,
        15,
        25,
        32,
        34,
        35,
        45,
        55,
        64,
        68,
        75,
        85,
        95,
        106,
        108,
        115,
        124,
    ]

    vertices_up = odom_ups[up_idxs]
    # vertices_up[:,1] += tolerance
    vertices_up[0, 0] -= 1.5 * tolerance_x
    vertices_up[-1, 0] += 1.5 * tolerance_x
    vertices_up[:, 1] -= tolerance_y * 1.1
    vertices_up[:, 1] -= tolerance_y * 1.1
    vertices_down = odom_downs[down_idxs]
    # vertices_down[:,1] -= tolerance
    vertices_down[0, 0] += 1.5 * tolerance_x
    vertices_down[-1, 0] -= 1.5 * tolerance_x
    vertices_down[:, 1] += tolerance_y * 1.1
    vertices_down[:, 1] += tolerance_y * 1.1
    vertices = np.vstack((vertices_up, vertices_down))
    print(vertices.shape)
    pick_area = Polygon(vertices)

    return pick_area


##########################################################
# Get pick area based on skeleton row map given tolerance
# The returned pick_area is in shapely polygon
# This polygon is in higher accuracy, but longer time to
# remove outliers
##########################################################
def get_pick_polygon(odom_NDs, odom_FDs, tolerance=5):
    # an hexagon for the map
    odom_downs = np.copy(odom_NDs)
    odom_ups = np.copy(odom_FDs)
    odom_downs[:, 1] += tolerance
    odom_ups[:, 1] -= tolerance
    vertice_pts = np.vstack((odom_ups, odom_downs[::-1]))
    print(vertice_pts.shape)
    polygon = Polygon(vertice_pts)

    return polygon


############################################################
# Remove repeated (or small changed) values from a 1d array
############################################################
def remove_static_values(one_d_array, threshold=0.2):
    delta_array = np.abs(np.diff(one_d_array))
    jump_idxs = np.where(delta_array > threshold)[0]
    #     print(delta_array)
    return jump_idxs


######################################################################
# Assign row ids to the dataset. The header of the data is as follows:
# Input: Nx12
#   ["GPS_TOW", "LAT", "LON", "HEIGHT", "ax", "ay", "az", "raw_mass", "east", "northing", "x", "y"]
# Input: Nx14
#   ["GPS_TOW", "LAT", "LON", "HEIGHT", "ax", "ay", "az", "raw_mass", "east", "northing", "x", "y", "row_x", "row_id"]


######################################################################
def assign_row_ids_DBSCAN(cart_data, bed_xs, show=False):  # used 2025
    # get row xs from bed xs
    row_xs = (bed_xs[:-1] + bed_xs[1:]) / 2.0
    # shrink time stamps
    odom_ts_xs = np.vstack(
        (cart_data[:, 0] / 100000, cart_data[:, -2])
    ).T  # used timestamp in one col and position in another as DBSCAN input

    clustering = DBSCAN(eps=0.3, min_samples=200).fit(odom_ts_xs)
    labels = clustering.labels_
    # DBSCAN results
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("number of clusters: ", n_clusters_)
    print("number of noise: ", n_noise_)
    if show:
        for i in set(labels):
            idx = np.where(labels == i)
            plt.scatter(cart_data[:, 0][idx] / 100000, cart_data[:, -2][idx], s=1)
            plt.xlabel("Stamps")
            plt.ylabel("row_xs")

    idx_good = np.where(labels != -1)[0]
    cart_data = cart_data[idx_good, :]
    labels = labels[idx_good]

    cart_data = np.hstack((cart_data, np.ones((len(cart_data), 2))))
    for i in set(labels):
        idxs = np.where(labels == i)
        xs = cart_data[idxs, -4]
        ys = cart_data[idxs, -3]
        # delete where delta y is so small
        move_idxs = remove_static_values(
            ys[0], threshold=0.2
        )  # remove if the points that has not moved 20 cm in y direction
        if len(move_idxs) > 0:
            mean_x = np.median(xs[0][move_idxs])
        else:
            mean_x = np.median(xs)
        row_id = np.argmin(np.abs(mean_x - row_xs))
        row_x = row_xs[row_id]
        cart_data[idxs, -2] = row_x
        cart_data[idxs, -1] = row_id

    row_nums = len(set(cart_data[:, -1]))
    print("number of rows: ", row_nums)

    return cart_data, row_nums, row_xs
