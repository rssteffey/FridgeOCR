#! /usr/bin/env
# -*- coding: utf-8 -*-

#Requires Python 2.7 to run
#(Because OpenCV is not a fan of Python 3)


#This file creates the training data through assisted learning
#The first half of the file is essentially identical to the fridgeTweet.py image processing part
#It should PROBABLY just be a shared function between the two... 
#but I kept altering my training environment to experiment with what different variable tweaks would do, so it was easier this way

#Also, there are plenty of magic numbers here.  I won't even apologize for that.  There are way too many values for me to properly label and utilize up top.
#On a personal script like this it's much easier for me to just find the proper context and change it

from twython import Twython
from operator import itemgetter
import cv2
import numpy as np
import difflib
import time

lastTweet = ''


#------OpenCV stuff-------

camera_port = 0
ramp_frames = 30
camera = cv2.VideoCapture(camera_port)
 
def get_image():
	retval, im = camera.read()
	return im

for i in xrange(ramp_frames):
 temp = get_image()
camera_capture = get_image()

height=265
width=180

crop = camera_capture[125:(125+height), 230:(230+width)]

toCropRect = np.array([
	 [0, 0],
	 [width - 1, 0],
	 [width - 1, height - 43],
	 [0, height - 1]], dtype = "float32")

dst = np.array([
	 [0, 0],
	 [width - 1, 0],
	 [width - 1, height - 1],
	 [0, height - 1]], dtype = "float32")
 
M = cv2.getPerspectiveTransform(toCropRect, dst)
warp = cv2.warpPerspective(crop, M, (width, height))

filer ="C:\\Users\\Owner\\Dropbox\\fridge\\images\\test_image.png"
cv2.imwrite(filer, warp)
filer ="C:\\Users\\Owner\\Dropbox\\fridge\\images\\test_full.png"
cv2.imwrite(filer, camera_capture)
del(camera)

#-----Begin Processing------

img = cv2.imread('C:\\Users\\Owner\\Dropbox\\fridge\\images\\test_image.png',cv2.CV_LOAD_IMAGE_COLOR)
colorblur = cv2.GaussianBlur(warp,(5,5),0)
img_hsv = cv2.cvtColor(colorblur, cv2.COLOR_BGR2HSV)

imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(imgray,(15,15),0)
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

boundaries = [
	([24, 220, 80], [34, 255, 255]), #yellow
	([50, 100, 50], [96, 255, 255]), #green
	([100, 130, 50], [125, 255, 255]), #blue
	([150, 210, 50], [180, 255, 255]), #Top reds
	([0, 210, 50], [15, 255, 255]) #Bottom reds
]

cv2.imwrite('C:\\Users\\Owner\\Dropbox\\fridge\\images\\hsv.png',colorblur)
mask = np.zeros((height,width,1), np.uint8)
count=0

for (lower, upper) in boundaries:
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")
 
	r1 = cv2.inRange(img_hsv, lower, upper)
	mask = cv2.add(r1,mask)
output = cv2.bitwise_and(colorblur, colorblur, mask = mask)
fileOut = 'C:\\Users\\Owner\\Dropbox\\fridge\\images\\combo.png'

cv2.imwrite(fileOut,output)

backtogray = cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(backtogray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

backtorgb = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
cv2.drawContours(backtorgb, contours, -1, (0,255,0), 1)

cv2.imwrite('C:\\Users\\Owner\\Dropbox\\fridge\\images\\contours.png',backtorgb)

#-----End duplicate segment-----


#-----Manual Input Training-----
#Loops over all contours found, highlighting the current one, and requiring the press of a keyboard key to correspond with the shape
#I found the classification data most effective after 3 or 4 times running this trainer (on differently arranged sets of all fridge magnets)

samples =  np.empty((0,400))
responses = []
keys = [i for i in range(97,123)]

for cnt in contours:
	rect = cv2.boundingRect(cnt)
	#if rect[2] < 30 or rect[3] < 30: continue
	if cv2.contourArea(cnt) > 100:
		print cv2.contourArea(cnt)
		[x,y,w,h] = cv2.boundingRect(cnt)

		if  h>10:
			cv2.rectangle(backtorgb,(x,y),(x+w,y+h),(0,0,255),2)
			roi = backtogray[y:y+h,x:x+w]
			roismall = cv2.resize(roi,(20,20))
			cv2.imshow('norm',backtorgb)
			key = cv2.waitKey(0)

			if key == 27:  # (escape to quit)
				sys.exit()
			elif key in keys:
				responses.append(key)
				sample = roismall.reshape((1,400))
				samples = np.append(samples,sample,0)
			print key

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"
for resp in responses:
	print resp
f_handle_response = file('generalresponses.data','a')
f_handle_sample = file('generalsamples.data','a')
np.savetxt(f_handle_sample,samples)
np.savetxt(f_handle_response,responses)
f_handle_response.close()
f_handle_sample.close()

#-----end input section-----