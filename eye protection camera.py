import cv2 as cv
import os

video = cv.VideoCapture(0)
target_path=os.path.join('/Users/sungwoo/Desktop/컴퓨터비전', "video.mov")
target_fourcc = 'H264'
fourcc= cv.VideoWriter_fourcc(*target_fourcc)
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
me = cv.imread('/Users/sungwoo/Desktop/컴퓨터비전/me.png')
image = cv.resize(me, (100, 100))

if video.isOpened() :
    target=cv.VideoWriter(target_path, fourcc, 20.0, (width, height))
    
    center = (300,300)
    radius = 50
    thickness = -1

    pause=False
    
    while True :
        valid, img = video.read()
        if not valid :
            break

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_region = img[y:y+h, x:x+w]
            resized = cv.resize(image, (w, h))
            face_region[:] = resized

            cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        target.write(img)

        color = (0,0,255)
        cv.circle(img, center, radius, color, thickness)

        cv.imshow('Frame', img)

        key=cv.waitKey(1)
        if key == 27 :
            break
        elif key == 32 :
            pause = not pause

        if pause :
            color = (0,255,0)
            cv.circle(img, center, radius, color, thickness)
            cv.imshow('Frame', img)

            while True :
                key = cv.waitKey(1)

                if key == 27 :
                    break
                elif key == 32 :
                    pause = not pause
                    break
                
        if(pause == True) :
            break

    target.release()
    video.release()
    cv.destroyAllWindows()