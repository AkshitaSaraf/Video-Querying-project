#some basic imports and setups
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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


# Playing video from file:

import time

#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()

cap = cv2.VideoCapture('animal.mp4')

try:
    if not os.path.exists('data11'):
        os.makedirs('data11')
except OSError:
    print ('Error: Creating directory of data')

currentFrame = 1000
c=0
totalframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
videotime = 1000*totalframe / fps
print(videotime*pow(10,-3))
timing = []
count2 = 0

while(c<8800):
    # Capture frame-by-frame
    #cap.set(cv2.CAP_PROP_POS_MSEC,2000)
    ret, frame = cap.read()
    if currentFrame%100==0:
        # Saves image of the current frame in jpg file
        name = './data11/frame' + str(currentFrame) + '.jpeg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        count2 = count2+1
        #cv2.imshow("20sec",frame)
        #cv2.waitKey()
    # To stop duplicate images
    currentFrame += 1
    c=c+1


frame_time = videotime / count2
timing.append(0)
for i in range (1,count2+1):
    t = timing[i-1] + frame_time
    timing.append(t)
    # When everything done, release the capture
for i in timing:
    print(i*pow(10,-3))
cap.release()
cv2.destroyAllWindows()
image_dir = os.path.join(current_dir, 'data11')



# get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]
# load all images
imgs = []


for f in img_files:
    imgs.append(cv2.imread(f))


count=0
for i, img in enumerate(imgs):
    count += 1

# plot images
fig = plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs):
    fig.add_subplot(1, count, i + 1)
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
# plt.show()


from alexnet import AlexNet
from caffe_classes import class_names

#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [1, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 1000, [])


#define activation of last layer as score
score = model.fc8

#create op to calculate softmax
softmax = tf.nn.softmax(score)

with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    model.load_initial_weights(sess)

    # Create figure handle
    fig2 = plt.figure(figsize=(15, 6))

    # Loop over all images
    imageId= 1000
    imageIds = []
    imageClassNames = []

    for i, image in enumerate(imgs):
        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (227, 227))

        # Subtract the ImageNet mean
        img -= imagenet_mean

        # Reshape as needed to feed into model
        img = img.reshape((1, 227, 227, 3))

        # Run the session and calculate the class probability
        probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        #print(probs)


        # Get the class name of the class with the highest probability
        class_name = class_names[np.argmax(probs)]
        imageIds.append(imageId)
        imageId += 100
        imageClassNames.append(class_name)

        #print(class_name)
        #
        # # Plot image with class name and prob in the title
        #fig2.add_subplot(1, count, i + 1)
        #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #plt.title("Class: " + class_name + ", probability: %.4f" % probs[0, np.argmax(probs)])
        #plt.axis('off')
        #plt.show()

    for index in range(0,len(imageIds)):
          print(imageIds[index],imageClassNames[index] )
    print("Enter query->")
    x = input()
    count1 = 0
    for i in range(0,count2+1):
         if x not in imageClassNames[i]:
               count1 = count1 + 1
               continue
         else:
               time2play=timing[count1+1]
               print(time2play)
               break


cap = cv2.VideoCapture('animal.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC,time2play)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("frame", gray)
        # & 0xFF is required for a 64-bit system
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()