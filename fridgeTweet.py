#! /usr/bin/env
# -*- coding: utf-8 -*-

#Requires Python 2.7 to run
#(Because OpenCV is not a fan of Python 3)

from twython import Twython
from operator import itemgetter
import cv2
import numpy as np
import difflib
import time

lastTweet = ''

while 1:
	#------OpenCV-------

	# Camera 0 seems to be the camera in most cases (Specifically my laptop and webcam on desktop)
	camera_port = 0
	#Number of frames to throw away while the camera adjusts to light levels
	ramp_frames = 30
	camera = cv2.VideoCapture(camera_port)
	 
	# Captures a single image from the camera and returns it in PIL format
	def get_image():
		retval, im = camera.read()
		return im
	# Ramp the camera and use the later frame
	for i in xrange(ramp_frames):
		temp = get_image()
	print("Taking image...")
	camera_capture = get_image()

	height=265
	width=180

	crop = camera_capture[125:(125+height), 230:(230+width)]
	
	toCropRect = np.array([
		 [0, 0],
		 [width - 1, 0],
		 [width - 1, height - 43],
		 [0, height - 1]], dtype = "float32")
	# construct our destination points which will map the screen to a top-down view
	dst = np.array([
		 [0, 0],
		 [width - 1, 0],
		 [width - 1, height - 1],
		 [0, height - 1]], dtype = "float32")
	 
	# calculate the perspective transform matrix and warp
	# the perspective to grab the screen
	#(This is because we don't have a shelf directly across from the fridge, so I've had to set-up at a very skewed angle)
	M = cv2.getPerspectiveTransform(toCropRect, dst)
	warp = cv2.warpPerspective(crop, M, (width, height))

	file1 ="C:\\Users\\Owner\\Dropbox\\fridge\\images\\test_image.png"
	cv2.imwrite(file1, warp)
	file2 ="C:\\Users\\Owner\\Dropbox\\fridge\\images\\test_full.png"
	cv2.imwrite(file2, camera_capture)
	del(camera)

	#-----Begin Processing Image------

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

	#Save color-blurred image (For human benefit in debugging)
	cv2.imwrite('C:\\Users\\Owner\\Dropbox\\fridge\\images\\hsv.png',colorblur)
	
	mask = np.zeros((height,width,1), np.uint8)
	count=0
	# loop over the boundaries
	for (lower, upper) in boundaries:
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
	 
		# find the colors within the specified boundaries and apply the mask
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
	#-----End Image Processing-----
	
	
	#-----Begin training-----
	# Data comes from previous assisted classification (in trainFridge.py) 
	samples = np.loadtxt('generalsamples.data',np.float32)
	responses = np.loadtxt('generalresponses.data',np.float32)
	responses = responses.reshape((responses.size,1))

	model = cv2.KNearest()
	model.train(samples,responses)
	#-----End Training-----

	
	
	#-----Begin Testing-----
	out = np.zeros(backtorgb.shape,np.uint8)
	outputter=''
	ordered=[]
	prevY = 0;
	for cnt in contours:
		#calculate centroid
		if cv2.contourArea(cnt)>50:
			M = cv2.moments(cnt)
			centroid_x = int(M['m10']/M['m00'])
			centroid_y = int(M['m01']/M['m00'])
			avgColor = img_hsv[centroid_y][centroid_x]
			centroid_y = int(centroid_y/10)
			#round to current row if close
			if centroid_y + 1 >= prevY and centroid_y - 1 <= prevY:
				centroid_y = prevY
			prevY = centroid_y

			ordered.append((cnt, centroid_x, centroid_y, avgColor))

	ordered = sorted(ordered, key=lambda x:x[1], reverse=True)
	ordered = sorted(ordered, key=lambda x:x[2], reverse=True)
	lasty = 9999
	for cnt,k,l, avg in ordered:

		string=''
		if cv2.contourArea(cnt)>50:
			[x,y,w,h] = cv2.boundingRect(cnt)
			if  h>5:
				cv2.rectangle(backtorgb,(x,y),(x+w,y+h),(0,255,0),2)
				roi = backtogray[y:y+h,x:x+w]
				roismall = cv2.resize(roi,(20,20))
				roismall = roismall.reshape((1,400))
				roismall = np.float32(roismall)
				retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 8)
				
				neighbors = neigh_resp.tolist()
				
				#Check all the color ranges established for best guess
				if avg[0] >= 45 and avg[0] < 95: #greens
					if chr(results[0][0]) in 'cgnrx':
						string = chr(results[0][0])
					else:
						for letter in neighbors[0]:
							if chr(int(letter)) in 'cgnrx':
								string = chr(int(letter))
								break;
				elif avg[0] >= 20 and avg[0] < 45: #yellows
					if chr(results[0][0]) in 'ahotz':
						string = chr(results[0][0])
					else:
						for letter in neighbors[0]:
							if chr(int(letter)) in 'ahotz':
								string = chr(int(letter))
								break;
				elif avg[0] >= 95 and avg[0] < 130: #blues
					if chr(results[0][0]) in 'elms':
						string = chr(results[0][0])
					else:
						for letter in neighbors[0]:
							if chr(int(letter)) in 'elms':
								string = chr(int(letter))
								break;
				elif (avg[0] >= 130 and avg[0] < 180) or (avg[0] >= 0 and avg[0] < 20): #pinkorangered
					if chr(results[0][0]) in 'bdfijkpquvwy':
						string = chr(results[0][0])
					else:
						for letter in neighbors[0]:
							if chr(int(letter)) in 'bdfijkpquvwy':
								string = chr(int(letter))
								break;
				
				#If somehow, the string matched none of these criteria (possible), then just go with the first guess
				if string == '':
						string = chr(results[0][0])
				
				#string = chr(results[0][0])
				#Check for spaces or new lines
				if lasty > l:
					outputter = ' ' + outputter
				elif lastx - k > 38:
					outputter = ' ' + outputter
				lasty = l;
				lastx = k;
				outputter = string + outputter
				cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

	#-----End Testing-----

	#-----Begin Final Tweet Prep-----
	
	#Get ratio of whitespace to number of characters (Higher ratio means it's probably a mistake)
	#[This happens when the kitchen lights get turned off and the camera data turns super noisy]
	outputter = outputter.upper()
	print outputter
	spaceCount = 0
	letterCount = 0
	gibberish=True
	for letter in outputter:
		if letter == ' ':
			spaceCount = spaceCount + 1
		else:
			letterCount = letterCount + 1
	if spaceCount == 0:
		gibberish=False
	elif(spaceCount/letterCount < .4):
		gibberish=False
	print 'Photo taken'
	
	#If the new tweet is different from our last one, AND not a random jumble of gibberish (Well, knowing my room mates that one could still happen)
	if ((difflib.SequenceMatcher(None, outputter, lastTweet).ratio()) < .7) and (len(outputter) > 1) and (not gibberish):

		toTweet = outputter

		#Twitter Variables (Replaced w/ dummies for GitHub posting)
		APP_KEY = 'AllAroundMeAreFamiliarFaces'
		APP_SECRET = 'WornOutPlacesWornOutFaces'
		OAUTH_TOKEN = 'BrightAndEarlyForTheirDailyRaces'
		OAUTH_TOKEN_SECRET = 'GoingNowhereGoingNowhere'

		#Twitter client
		twitter = Twython(APP_KEY, APP_SECRET,
						  OAUTH_TOKEN, OAUTH_TOKEN_SECRET)

		lastTweet = outputter
		try:
			twitter.update_status(status=toTweet)
		except:
			print 'Tweet failed.  Either duplicate post or network error.'
		
		print 'Tweeted: ' + lastTweet
	time.sleep(60)