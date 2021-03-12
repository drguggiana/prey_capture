import paths
import cv2
import numpy as np
import os
import random

# define the number of frames to extract
target_frames = 100

# define the target folder
target_variable = 'VR'

if target_variable == 'Video':
    target_folder = paths.videoexperiment_path
else:
    target_folder = paths.vrexperiment_path

# get the sample of paths
video_sample = random.sample([el for el in os.listdir(target_folder) if '.avi' in el], target_frames)

# get a random set of video paths from the target folder
for idx, video in enumerate(video_sample):

    # create the video object
    cap = cv2.VideoCapture(os.path.join(target_folder, video))

    # read the first image
    img = cap.read()[1]

    # release the video file
    cap.release()

    # print the image to the target path
    cv2.imwrite(os.path.join(paths.corner_path, target_variable, 'pics', target_variable+str(idx)+'.jpg'), img)
