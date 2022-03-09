import tifffile as tiff
import os


# define the path to the target file
target_file = r"C:\temp_minian"
# define the target video folder
target_output = r'D:\DeepWonder_test\Data'

file_path = os.path.join(target_output, 'minian_mc.tif')
# file_path = r"D:\DeepWonder_test\191_4_2_2000.tif"
im = tiff.imread(file_path)

im2 = tiff.TiffFile(file_path)
print('yay')
