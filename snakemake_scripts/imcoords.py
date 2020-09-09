import cv2
import matplotlib.pyplot as plt

path = r'J:\Drago Guggiana Nilo\Prey_capture\Corner_pics\VR\10.jpg'

im = cv2.imread(path, 0)
plt.plot(im)