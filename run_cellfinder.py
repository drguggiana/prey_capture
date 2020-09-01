# cellfinder -s "F:\Test_immuno\Green" -b "F:\Test_immuno\Red" -o "F:\Test_immuno\Out" -x 1 -y 1 -z 5 --register --no-detection --no-classification --no-standard-space --orientation coronal --affine-n-steps 2
#
# cellfinder -s "F:\Test_immuno\Green" -b "F:\Test_immuno\Red" -o "F:\Test_immuno\Out" -x 1 -y 1 -z 5 --register --no-standard-space --orientation coronal --affine-n-steps 2

# cellfinder -s "F:\Test_immuno\Green" -b "F:\Test_immuno\Red" -o "F:\Test_immuno\Out" -x 1 -y 1 -z 5 --register --no-standard-space --orientation coronal --affine-n-steps 2 --download-path "D:\Temp"

# cellfinder -s "F:\Test_immuno\Green" -b "F:\Test_immuno\Red" -o "F:\Test_immuno\Out" -x 1 -y 1 -z 5 --register --orientation "coronal" --affine-n-steps 2 --download-path "D:\Temp"
#
# amap "F:\Test_immuno\Red" "F:\Test_immuno\Out2" -x 1 -y 1 -z 5 --orientation coronal --affine-n-steps 2

# amap "F:\Test_immuno\Red" "F:\Test_immuno\Out3" -x 1 -y 1 -z 5 --orientation coronal --affine-n-steps 3 --download-path "D:\Temp" --install-path "D:\.cellfinder"

# amap "F:\Test_immuno\Red_slice2" "F:\Test_immuno\Out_slice2" -x 1 -y 1 -z 5 --orientation coronal --affine-n-steps 2 --download-path "D:\Temp" --install-path "D:\.cellfinder"

# cellfinder -s "F:\Test_immuno\Green" -b "F:\Test_immuno\Red" -o "F:\Test_immuno\Out_cf" -x 1 -y 1 -z 5 --register --orientation "coronal" --affine-n-steps 2 --download-path "D:\Temp" --install-path "D:\.cellfinder"


import pandas as pd

# a = pd.read_hdf(r'X:\Isa\IG_181102c-2-PR1-0004DLC_resnet50_PupRetrievalMay20shuffle1_100000_sk.h5')
a = pd.read_hdf(r"X:\Isa\IG_180914a-2-PR2-0007croppedDLC_resnet50_PupRetrievalTesAug27shuffle1_50000_bx.h5")

# a = pd.read_pickle(r'X:\Isa\IG_181102c-2-PR1-0004DLC_resnet50_PupRetrievalMay20shuffle1_100000_full.pickle')

# print(a['single', 'LLcornerArena', 'y'])

print('yay')