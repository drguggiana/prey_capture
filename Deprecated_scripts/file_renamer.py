import os
import glob
import re

# Simple script to rename files. This was used to fix file names before a convention was decided upon.


# base_path = r"C:\Users\setup\Documents\VR_files"
base_path = r"J:\Drago Guggiana Nilo\Prey_capture\VRExperiment"
all_files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]

# Renaming 15 July 2020
# files_to_change = [f for f in all_files if "succ" not in f and "fail" not in f]
# for file in files_to_change:
#     if "test" in file:
#         parts = file.rsplit("_test")
#         if parts[-1][0] == ".":
#             pass
#     else:
#         parts = re.findall("(.+DG_200526_.)(.+)", file)
#         new_name = parts[0][0] + "_test_" + parts[0][1]
#         os.rename(os.path.join(base_path, file), os.path.join(base_path, new_name))

# Renaming 16 July 2020
# ref_flags = ["VPrey"]
# for ref in ref_flags:
#     files_to_reference = [f for f in all_files if ref in f]
#     for file in files_to_reference:
#         # Find dates and match closest files
#         file_datetime = file.split("sync"+ref)[0]
#         match_files = glob.glob(os.path.join(base_path, file_datetime+"*"))
#         for m in match_files:
#             if ref not in m:
#                 parts = m.split(file_datetime)
#                 new_name = parts[0] + file_datetime + ref + "_" + parts[-1]
#                 if m != new_name:
#                     os.rename(m, new_name)
#                 else:
#                     pass

ref_flags = ["VPrey"]
for ref in ref_flags:
    files_to_reference = [f for f in all_files if ref in f]
    for file in files_to_reference:
        # Find dates and match closest files
        if ("succ" in file) and ("succ_real" not in file):
            new_name = file.replace("succ", "succ_real")
            os.rename(os.path.join(base_path, file), os.path.join(base_path, new_name))
