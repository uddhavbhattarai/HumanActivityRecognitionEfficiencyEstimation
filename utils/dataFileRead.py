import argparse
import numpy as np
import time
from math import *
import pymap3d
import pandas
import utm
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta


CSV_HEAD = [
    "rpi_utc_time",
    "gps_utc_time",
    "GPS_TOW",
    "LAT",
    "LON",
    "HEIGHT",
    "ax",
    "ay",
    "az",
    "raw_mass",
]
USED_HEAD = ["GPS_TOW", "LAT", "LON", "HEIGHT", "ax", "ay", "az", "raw_mass"]


# read the files from the specified directory
def read_files(folder_dir):
    # get the list of files from the folder
    directory = os.path.abspath(folder_dir)
    raw_data = {}
    # read all the files from the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                file_key = file[:-4]
                print("*****converting " + file + " into numpy array****")
                sd_data_csv = pandas.read_csv(
                    file_path,
                    on_bad_lines="error",
                    usecols=USED_HEAD,
                    encoding="ISO-8859-1",
                )
                sd_raw_data = convert2array(sd_data_csv)
                if len(sd_raw_data) > 0:
                    raw_data[file_key] = sd_raw_data
                print("************************************")

    return raw_data


def convert2array(sd_data_csv, min_lat=36, max_lat=37, min_lon=-122, max_lon=-121):
    sd_data = sd_data_csv.values
    print(sd_data.shape)
    sd_data_raw = []

    for i in range(len(sd_data)):
        data_line = list(sd_data[i, :])
        if len(data_line) == 8:
            try:
                data_line = list(sd_data[i, :])
                data_line = list(map(float, data_line))
                # if data_line[1] > min_lat and data_line[1] < max_lat and data_line[2] < max_lon and data_line[2] > min_lon:
                e, n, _, _ = utm.from_latlon(data_line[1], data_line[2])
                data_line += [e, n]
                sd_data_raw.append(data_line)
            #             print(e,n)
            except Exception as error:
                print(error)
                print("error line: ", data_line)
    sd_data_raw = np.array(sd_data_raw)
    print(sd_data_raw.shape)

    return sd_data_raw


def dict2npy(py_dict, file_name="dict_data.npy"):

    np.save(file_name, py_dict)


def npy2dict(npy_file):

    py_dict = np.load(npy_file, allow_pickle=True).item()

    return py_dict


def save_yield_file(data_folder, data_time, yield_data):
    data_path = os.path.join(data_folder, data_time)
    npy_file = data_path + "/" + data_time + "_yield.npy"
    dict2npy(yield_data, npy_file)

    return


def save_yield_file_tray_id(data_folder, data_time, yield_data):
    data_path = os.path.join(data_folder, data_time)
    npy_file = data_path + "/" + data_time + "_yield_tray_id.npy"
    dict2npy(yield_data, npy_file)

    return


def load_yield_data(data_folder, data_time):
    data_path = data_folder + data_time
    npy_file = data_path + "/" + data_time + "_yield.npy"
    data_dict = npy2dict(npy_file)

    return data_dict


def load_yield_data_tray_id(data_folder, data_time):
    data_path = data_folder + data_time
    npy_file = data_path + "/" + data_time + "_yield_tray_id.npy"
    data_dict = npy2dict(npy_file)

    return data_dict


def load_raw_data(folder_path, data_date, in_bound_data=False):
    # date obtained from input args
    data_path = folder_path + data_date
    npy_file = data_path + "/" + data_date + ".npy"
    in_bound_npy_file = data_path + "/" + data_date + "_inbound.npy"
    if in_bound_data and os.path.exists(in_bound_npy_file):
        raw_data = npy2dict(in_bound_npy_file)
    else:
        # check if the npy exists
        if os.path.exists(npy_file):
            raw_data = npy2dict(npy_file)
        else:
            raw_data = read_files(data_path)
            print("read csv file", raw_data.keys())
            dict2npy(raw_data, file_name=npy_file)

    return raw_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", "-s", action="store_true")
    parser.add_argument("--date", "-d", dest="date", default="70921", type=str)

    args = parser.parse_args()
    data_folder = "/home/afo_1332/Documents/carritoData/arduino_cart/"
    raw_data = load_raw_data(data_folder, args.date)

    if args.show:
        dn = 10
        plt.figure()
        plt.rcParams["figure.figsize"] = [20, 10]
        for cart in raw_data.keys():
            try:
                N = len(raw_data[cart][:, 0])
                print("N is: ", N)
                max_id = np.argmax(np.diff(raw_data[cart][:, 0]))
                print("delta t mean: ", np.diff(raw_data[cart][:, 0]).mean())
                print("delta t max: ", np.diff(raw_data[cart][:, 0])[max_id])
                print(
                    "delta t over 100ms: ",
                    len(np.where(np.diff(raw_data[cart][:, 0]) > 100)[0]) / N,
                )
                plt.scatter(raw_data[cart][0:N:dn, 2], raw_data[cart][0:N:dn, 1], s=1)
            except Exception as error:
                print(raw_data[cart].shape)
                print(error)

        plt.show()
