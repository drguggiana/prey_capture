#!/usr/bin/env python3
import os
import re
import glob


def rename_social(file_paths):
    """
    Used to rename old social prey capture files
    """
    # Set up a counter
    file_counter = 0
    # make a list of failed files
    failed_paths = []
    # for each file path
    for file in file_paths:
        # check if the file is there
        if not os.path.isfile(file):
            failed_paths.append('_'.join(('old', file)))
            continue

        # Parse the old file name to check if it is a social experiment or not
        animalnameRegex = re.compile(r'[A-Z][A-Z]_\d\d\d\d\d\d_[a-z]')
        animals = animalnameRegex.findall(file)

        if len(animals) > 1:
            # This is a social prey capture test
            first_animal = animals[0]
            # Split the file
            parts = file.split(first_animal)
            # Check to see if previously modified, otherwise make the modification
            if "social" in file:
                continue
            else:
                mod = 'social_' + first_animal
                new_path = "".join([parts[0], mod, parts[-1]])
        else:
            continue

        # check if the new path exists
        if os.path.isfile(new_path):
            failed_paths.append('_'.join(('new', new_path)))
            continue
        # change the file_name
        os.rename(file, new_path)
        # update the counter
        file_counter += 1

    print("_".join(("Total original files: ", str(len(file_paths)), "Successfully renamed files: ", str(file_counter))))
    return failed_paths


if __name__ == "__main__":

    for dir_path in [r"J:\Matthew McCann\prey_capture\bonsai_out",
                     r"J:\Matthew McCann\prey_capture\sync_data",
                     r"J:\Matthew McCann\prey_capture\motive_out"
                     ]:

        file_paths = glob.glob(dir_path + os.sep + "*")
        rename_social(file_paths)