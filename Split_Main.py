#Main function for image splitting
#author: ToInoJo

import numpy as np
import os
#import SimpleCV as simp
from PIL import Image
import scipy
from pathlib import Path
from itertools import product
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import scipy
from scipy import signal
import tkinter as tk
from tkinter import filedialog
import cv2
import time

def tile(filename, impath, dir_in, dir_out, ver,hor):
    name, ext = os.path.splitext(filename)
    img = Image.open(impath)
    w, h = img.size
    
    grid = product(range(0, h-h%ver, ver), range(0, w-w%hor, hor))
    for i, j in grid:
        box = (j, i, j+ver, i+hor)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)


############# Image loading ############

mode = 0    #0 for one image, 1 for the whole folder
splitmode = 0 #0 for basic grid, 1 for advanced
impath = Path(r"C:\\Users\\pawzi\\Documents\\GitHub\\Split-picture-into-parts\\Images\\T5.jpg")
dirpath = Path(r"C:\\Users\\pawzi\\Documents\\GitHub\\Split-picture-into-parts\\Images")

#im = Image.open(impath)
curr = os.getcwd()

filename = filedialog.askopenfilename(initialdir=curr,title='Open file to encrypt')
im = Image.open(filename)

############# Image Splitting ############

horizontal_no = 20
vertical_no = 12

t_start = time.time()
image_for_cutting = Image.open(filename)
imge = cv2.imread(filename)
#greyscale_image = imge.convert('L')
img_raw = np.asarray(imge)
t_read = time.time()
t_print = t_read - t_start
print("Read time:", t_print, "seconds")
#img = scipy.ndimage.minimum_filter(img_raw, 44, mode = 'wrap')
#img = imge.filter(ImageFilter.MinFilter(size = 33))
size = (80, 160)
shape = cv2.MORPH_ELLIPSE
kernel = cv2.getStructuringElement(shape, size)
img = cv2.erode(imge, kernel)
t_filter = time.time()
t_print = t_filter - t_read
print("Filter time:", t_print, "seconds")

w, h, dump = img.shape
print("Picture dimensions: ", w, h)
brightness_vect = []

for j in range(1,w):
    b = 0
    for i in range(1,h):
        #pixelRGB = img.getpixel((j,i))
        pixelRGB = (img[j, i])
        R,G,B = pixelRGB
        b = (sum([R,G,B])/3) + b
    brightness_vect.append(b)

br_raw = np.array(brightness_vect)
#br_smooth = scipy.ndimage.minimum_filter(br_raw, 33, mode = 'mirror')
#br_smooth = scipy.ndimage.minimum_filter(br_raw, 3, mode = 'wrap')


brightness_vect1 = []

for j in range(1,h):
    b = 0
    for i in range(1,w):
        pixelRGB = (img[i, j])
        R,G,B = pixelRGB
        b = (sum([R,G,B])/3) + b
    brightness_vect1.append(b)
br1_raw = np.array(brightness_vect1)
#br_smooth = scipy.ndimage.minimum_filter(br_raw, 33, mode = 'mirror')
#br1_smooth = scipy.ndimage.minimum_filter(br1_raw, 3, mode = 'wrap')
t_summing = time.time()
t_print = t_summing - t_filter
print("Summing time:", t_print, "seconds")

br_smooth = scipy.signal.savgol_filter(br_raw, 60, 1, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)
br1_smooth = scipy.signal.savgol_filter(br1_raw, 60, 1, deriv=0, delta=1.0, axis=- 1, mode='interp', cval=0.0)

peaks_h,dump1 = scipy.signal.find_peaks(br_smooth, height=None, threshold=None, distance=100, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
peaks_w,dump2 = scipy.signal.find_peaks(br1_smooth, height=None, threshold=None, distance=100, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)



peaks_h = np.insert(peaks_h, 0, 0, axis=0)
peaks_h = np.insert(peaks_h, len(peaks_h), w, axis=0)
peaks_w = np.insert(peaks_w, 0, 0, axis=0)
peaks_w = np.insert(peaks_w, len(peaks_w), h, axis=0)
print("Peaks 1: ", peaks_h)
print("Peaks 2: ", peaks_w)

dir_out = filedialog.askdirectory(initialdir=curr,title='Open directory for saving img')
ogdir, filename = os.path.split(filename)
name, ext = os.path.splitext(filename)
for j in range(0, len(peaks_h)-1):
    for i in range(0, len(peaks_w)-1):
        box = (peaks_w[i], peaks_h[j], peaks_w[i+1], peaks_h[j+1])
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')
        image_for_cutting.crop(box).save(out)


#tobinimg = cv2.imread(filename)
#grayed = cv2.cvtColor(tobinimg, cv2.COLOR_BGR2GRAY)
#threshold,binarisedimg = cv2.threshold(grayed,0,255,(cv2.THRESH_BINARY+cv2.THRESH_OTSU))
#cv2.imshow("Binarised",binarisedimg)


#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#orb = cv2.ORB_create()
#kp = orb.detect(gray,None)
# compute the descriptors with ORB
#kp, des = orb.compute(gray, kp)
#fin=cv2.drawKeypoints(gray,kp,img)
#cv2.imwrite('orb_keypoints.jpg',img)
#GOAL: 7 cuts on width, 6 cuts on height

#params = cv2.SimpleBlobDetector_Params()

#detector = cv2.SimpleBlobDetector_create(params)
#keypoints = detector.detect(gray)

#blank = np.zeros((1, 1))
#blobs = cv2.drawKeypoints(gray, keypoints, blank)

#cv2.imshow("Blobs Using Area", blobs)
#cv2.imshow("Blobs Using Area",blobs)
#cv2.imwrite('blob.jpg',blobs)


plt.plot(br_raw)
plt.plot(br_smooth)
plt.show()
plt.plot(br1_raw)
plt.plot(br1_smooth)
plt.show()





