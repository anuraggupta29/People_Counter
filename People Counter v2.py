import numpy as np
import cv2 as cv
import Person
import time
import datetime
#from picamera.array import PiRGBArray
#from picamera import PiCamera

try:
    log = open('log.txt',"w")
except:
    print( "Unable to open log file.")

#COUNT METER
cnt_up   = 0
cnt_down = 0

#FETCH THE VIDEO
cap = cv.VideoCapture('Test Files/TestVideo.mp4')
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
line_down   = int(height*3/5)

up_limit =   int(height*1/5)
down_limit = int(height*4/5)

#BOTTOM SUBTRACTOR
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = True)

#STRUCTURING ELEMENTS FOR MORPHOGRAPHIC FILTERS
kernelOp = np.ones((3,3),np.uint8)
kernelOp2 = np.ones((5,5),np.uint8)
kernelCl = np.ones((11,11),np.uint8)

#VARIABLES
font = cv.FONT_HERSHEY_DUPLEX
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
        ret,imBin= cv.threshold(fgmask,200,255,cv.THRESH_BINARY)
        ret,imBin2 = cv.threshold(fgmask2,200,255,cv.THRESH_BINARY)

        #OPENING (ERODE-> DILATE) TO REMOVE NOISE.
        mask = cv.morphologyEx(imBin, cv.MORPH_OPEN, kernelOp)
        mask2 = cv.morphologyEx(imBin2, cv.MORPH_OPEN, kernelOp)

        #CLOSING (DILATE -> ERODE) TO JOIN WHITE REGIONS.
        mask =  cv.morphologyEx(mask , cv.MORPH_CLOSE, kernelCl)
        mask2 = cv.morphologyEx(mask2, cv.MORPH_CLOSE, kernelCl)
    except:
        print('EOF')
        print( 'UP:',cnt_up)
        print ('DOWN:',cnt_down)
        break

    #RETR_EXTERNAL RETURNS ONLY EXTREME OUTER FLAGS. ALL CHILD CONTOURS ARE LEFT BEHIND.
    contours = cv.findContours(mask2,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > areaTH:
            #IT REMAINS TO ADD CONDITIONS FOR MULTI-PEOPLE, OUTPUTS AND SCREEN INPUTS.
            M = cv.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv.boundingRect(cnt)

            new = True
            if cy in range(up_limit,down_limit):
                for i in persons:
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h: #THIS OBJECT IS CLOSE TO THE ONE DETECTED BEFORE
                        new = False
                        i.updateCoords(cx,cy) #UPDATE COORDINATES OF OBJECT AND RESET AGE

                        if i.going_UP(line_down,line_up) == True:
                            cnt_up += 1;
                            print( "ID:",i.getId(),'crossed going up at',time.strftime("%c"))
                            log.write("ID: "+str(i.getId())+' crossed going up at ' + time.strftime("%c") + '\n')

                        elif i.going_DOWN(line_down,line_up) == True:
                            cnt_down += 1;
                            print( "ID:",i.getId(),'crossed going down at',time.strftime("%c"))
                            log.write("ID: " + str(i.getId()) + ' crossed going down at ' + time.strftime("%c") + '\n')

                        break

                    if i.getState() == '1':
                        if i.getDir() == 'down' and i.getY() > down_limit:
                            i.setDone()

                        elif i.getDir() == 'up' and i.getY() < up_limit:
                            i.setDone()

                    if i.timedOut():
                        index = persons.index(i)
                        persons.pop(index) #REMOVE i FROM THE PERSONS LIST
                        del i     #FREE THE MEMORY OF I

                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1

            cv.circle(frame,(cx,cy), 5, (0,0,255), -1)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    #END OF [for cnt in contours]

    #DRAW PATH
    for i in persons:
        cv.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv.LINE_AA)

    #ADD CONTENT ON IMAGES
    str_up = 'UP: '+ str(cnt_up)
    str_down = 'DOWN: '+ str(cnt_down)

    cv.line(frame,(0, line_up),(width, line_up),(255,0,0),2)
    cv.line(frame,(0, line_down),(width,line_down),(0,0,255),2)
    cv.line(frame,(0, up_limit),(width,up_limit),(255,255,255),1)
    cv.line(frame,(0, down_limit),(width,down_limit),(255,255,255),1)

    cv.rectangle(frame,(0,line_up-20),(100,line_up),(255,0,0),-1)
    cv.rectangle(frame,(0,line_down-20),(100,line_down),(0,0,255),-1)
    cv.putText(frame, str_up ,(8,line_up-4),font,0.5,(255,255,255),1,cv.LINE_AA)
    cv.putText(frame, str_down ,(8,line_down-4),font,0.5,(255,255,255),1,cv.LINE_AA)

    cv.imshow('Frame',frame)
    cv.imshow('Mask',mask)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
#END OF [while(cap.isOpened())]

#CLEANING
log.flush()
log.close()
cap.release()
cv.destroyAllWindows()
endtime = datetime.datetime.now().replace(microsecond=0)
print('End Time : ', endtime)
print('Total time taken is : ', endtime-starttime)
