
'''
Using OpenCV takes a mp4 video and produces a number of images.
Requirements
----
You require OpenCV 3.2 to be installed.
Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py
Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

# Playing video from file:
import time


cap = cv2.VideoCapture('animal.mp4')

try:
    if not os.path.exists('data11'):
        os.makedirs('data11')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 0
c=0
totalframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
videotime = 1000*totalframe / fps
print(videotime)
timing = []

while(c<10000):

    # Capture frame-by-frame
    #cap.set(cv2.CAP_PROP_POS_MSEC,2000)
    ret, frame = cap.read()
    if currentFrame%100==0:
        starttime = time.time()
        timing.append(starttime)
        # Saves image of the current frame in jpg file
        name = './data11/frame' + str(currentFrame) + '.jpeg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        #cv2.imshow("20sec",frame)
        #cv2.waitKey()
    # To stop duplicate images
    currentFrame += 1
    c=c+1
# When everything done, release the capture
for i in timing:
    print(i)
cap.release()
cv2.destroyAllWindows()
