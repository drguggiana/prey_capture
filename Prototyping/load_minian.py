import sys
import paths
import dask
import functools as fct
import numpy as np
from tifffile import TiffWriter
import tifffile
import os
import cv2
from PIL import Image

minian_path = paths.minian_path
sys.path.append(minian_path)

from minian.utilities import open_minian
from minian.visualization import write_video
from minian.utilities import custom_arr_optimize


# define the path to the target file
target_file = r"C:\temp_minian"
# define the target video folder
target_output = r'D:\DeepWonder_test\Data'

# load the experiment
motion_corrected_data = open_minian(target_file)
# write the video
# write_video(motion_corrected_data.Y_fm_chk, "minian_mc.mp4", target_output)
arr = motion_corrected_data.Y_fm_chk

arr_opt = fct.partial(
    custom_arr_optimize, rename_dict={"rechunk": "merge_restricted"}
)
with dask.config.set(array_optimize=arr_opt):
    arr = arr.astype(np.float32)
    arr_max = arr.max().compute().values
    arr_min = arr.min().compute().values
den = arr_max - arr_min
arr -= arr_min
arr /= den
arr *= 255
arr = arr.clip(0, 255).astype(np.uint8)
w, h = arr.sizes["width"], arr.sizes["height"]

count = 0
full_out = os.path.join(target_output, 'minian_mc.tif')
pics = []
# with TiffWriter(os.path.join(target_output, 'minian_mc.tif'), bigtiff=False, imagej=True) as tif:
for blk in arr.data.blocks:
    # tifffile.imwrite(full_out, blk, append=True)
    # tif.write(blk)
    # tif.save(blk)
    # blk = np.array((blk - blk.min())/(blk.max() - blk.min())*255)
    # cv2.imwrite(full_out, blk)
    pics.append(np.array(blk))
    count += 1
    if count == 24:
        break

pics = np.concatenate(pics)

tifffile.imwrite(full_out, pics)
        # if idx == 0:
        #     tif.save(blk, description=str({'shape': [100, w, h]}))
        # else:
        # for frame in blk:
        #     tif.save(frame)
        # count += 1
        # if count == 2:
        #     break
print('yay')
