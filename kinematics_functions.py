import numpy as np


def wrap(angles):
    """wrap angles to the range 0 to 360 deg"""
    # modified from https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
    return angles % 360


def unwrap(angles):
    """unwrap angles in degrees"""
    return np.rad2deg(np.unwrap(np.deg2rad(angles)))


def heading_calculation(data_target, data_origin):
    """Calculate the heading angle of vectors, taking in the vector origin and the end"""
    heading_vector = np.rad2deg(np.array([np.arccos(point[0] / np.linalg.norm(point)) if point[1] > 0 else
                                - np.arccos(point[0] / np.linalg.norm(point)) for point in
                                (data_target - data_origin)]))
    return heading_vector


def reverse_heading(heading, data_origin, vector_length=0.1):
    """Calculate the vector end coordinates based on heading and vector origin"""
    data_target = np.array([[np.cos(angle)*vector_length, np.sin(angle)*vector_length] for angle in
                           np.deg2rad(heading)] + data_origin)
    return data_target


def bin_angles(angles, number_angles=36):
    """Bin angles for polar plots"""
    # get the angle bins
    angle_bins = np.arange(0, 360, 360 / (number_angles + 1))
    digitized_array = np.digitize(angles, angle_bins)
    # calculate the bin values
    binned_array = np.array([np.sum(digitized_array == el + 1) for el in np.arange(number_angles)])
    polar_coord = np.concatenate((angle_bins[:-1].reshape((-1, 1)), binned_array.reshape((-1, 1))), axis=1)
    return polar_coord


def distance_calculation(data_1, data_2):
    """Calculate the euclidean distance between corresponding points in the 2 input data sets"""
    return np.array([np.linalg.norm(point_1 - point_2) for point_1, point_2 in zip(data_1, data_2)])
