import doric as dr
import paths
import numpy as np
from skimage import io


def extract_doric(input_path):
    """Extract the contents of a .doric file"""

    data = dr.ExtractDataAcquisition(input_path)
    return data


def save_tif(input_data, output_path, normalize=False):
    """Save a numpy array into an 8bit multipage BIGtiff"""
    if normalize:
        # scale the output to max and turn into uint8 (for MiniAn)
        max_value = np.max(input_data)
        # for all the frames
        for idx, frames in enumerate(input_data):
            input_data[idx, :, :] = ((frames/max_value)*255).astype('uint8')
    # save the final stack
    io.imsave(output_path, input_data, bigtiff=True)
    return


def convert_doric_to_tif(input_path, output_path):
    """Convert a .doric file into a .tif file"""
    # load the file
    data = extract_doric(input_path)
    data = data[0]['Data'][0]['Data']
    # permute to leave frames as the first dimension
    data = np.transpose(data, [2, 0, 1])
    # save the tif file to the target location
    save_tif(data, output_path)

    return


if __name__ == "__main__":
    from glob import glob
    from os.path import isfile

    doric_list = glob(r'J:\Drago Guggiana Nilo\Prey_capture\VRExperiment\*.doric')

    # load the sample data
    # test_data = extract_doric(paths.doric_sample)
    # test_data = test_data[0]['Data'][0]['Data']

    for doric_file in doric_list:

        # get the output path
        # out_path = paths.doric_sample.replace('.doric', '.tif')
        out_path = doric_file.replace('.doric', '.tif')
        print("Checking if {} is already converted...".format(doric_file))

        if not isfile(out_path):
            print("Converting {} to .tif!".format(doric_file))
            # convert the file if it hasn't been converted already
            convert_doric_to_tif(doric_file, out_path)
            print("Converted!\n")

        else:
            print("File exists!\n")

    print('yay')
