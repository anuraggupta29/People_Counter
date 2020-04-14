import numpy as np
import cv2
import Person
import time
import datetime
#from picamera.array import PiRGBArray
#from picamera import PiCamera

#COUNT METER
cnt_up   = 0
cnt_down = 0

#FETCH THE VIDEO
cap = cv2.VideoCapture('Test Files/TestVideo.mp4')
#camera = PiCamera()
#camera.resolution = (160,120)
#camera.framerate = 5
#rawCapture = PiRGBArray(camera, size=(160,120))
#time.sleep(0.1)

starttime = datetime.datetime.now().replace(microsecond=0)
print('Start Time : ', starttime)

#GET FRAME DIMENSTIONS AND SET AREA THRESHOLD FOR PERSON SIZE
width = int(cap.get(3))
height = int(cap.get(4))
print('Resolution is : ({}, {})'.format( width, height))

frameArea = height*width
areaTH = frameArea/200
print('Area Threshold : ', areaTH)

#LINE MARKERS TO TRACK PEOPLE
line_up = int(height*2/5)
line_down = int(height*3/5)

up_limit =   int(height*1/5)
down_limit = int(height*4/5)

#BOTTOM SUBTRACTOR
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#STRUCTURING ELEMENTS FOR MORPHOGRAPHIC FILTERS
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#VARIABLES
font = cv2.FONT_HERSHEY_DUPLEX
persons = []
max_p_age = 5
pid = 1

while(cap.isOpened()):
    ret, frame = cap.read()

    for i in persons:
        i.age_one()

    #APPLY BACKGROUND SUBTRACTION
    fgmask = fgbg.apply(frame)
    fgmask2 = fgbg.apply(frame)

    #BINARIZATION TO REMOVE SHADOWS (GRAY COLOR)
    try:
        ret,imBin= cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2 = cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)

        #OPENING (ERODE-> DILATE) TO REMOVE NOISE.
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask2 = cv2.morphologyEx(imBin2, cv2.MORPH_OPEN, kernelOp)

        #CLOSING (DILATE -> ERODE) TO JOIN WHITE REGIONS.
        mask =  cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernelCl)

    except:
        print( 'UP:',cnt_up)
        print ('DOWN:',cnt_down)
        break

    #RETR_EXTERNAL RETURNS ONLY EXTREME OUTER FLAGS. ALL CHILD CONTOURS ARE LEFT BEHIND.
    contours = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > areaTH:
            #IT REMAINS TO ADD CONDITIONS FOR MULTI-PEOPLE, OUTPUTS AND SCREEN INPUTS.
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            if y in range(up_limit,down_limit):
                for index,i in enumerate(persons):
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h: #THIS OBJECT IS CLOSE TO THE ONE DETECTED BEFORE
                        new = False
                        i.updateCoords(cx,cy) #UPDATE COORDINATES OF OBJECT AND RESET AGE

                        if i.going_UP(line_down,line_up) == True:
                            cnt_up += 1
                            print('UP : {}, DOWN : {}'.format(cnt_up, cnt_down))

                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1
                            print('UP : {}, DOWN : {}'.format(cnt_up, cnt_down))

                        break

                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()

                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()

                    if i.timedOut():
                        persons.pop(index) #REMOVE i FROM THE PERSONS LIST
                        del i     #FREE THE MEMORY OF I

                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1

    #END OF [for cnt in contours]

#END OF [while(cap.isOpened())]

#CLEANING
#log.flush()
#log.close()
cap.release()
endtime = datetime.datetime.now().replace(microsecond=0)
print('End Time : ', endtime)
print('Total time taken is : ', endtime-starttime)
