import cv2
from fpdf import FPDF
import os
import numpy as np

url = "http://192.168.137.96:8080/video"
cap = cv2.VideoCapture(url)
ret = True
f1 = 0
i = 0

cap.set(3,320)
cap.set(4,240)
while ret:
    ret, frame = cap.read()
    if f1 == 0:
        print("press 's' to scan the document")
        print("press 'q' to quit")
        f1 = f1 + 1
    cv2.imshow("camera feed", frame)
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.destroyWindow("camera feed")
        cv2.imshow("Scanned photo", frame)
        print("press o to see original frame")
        print("press u if its unreadable")
        print("press b to convert it to black and white form")
        print("press e to see different effects")
        k1 = cv2.waitKey(0)
        if k1 == ord('o'):
            cv2.destroyWindow('Scanned photo')
            cv2.imwrite("E://pdf//scanned%d.jpg" % i, frame)
            i = i + 1
            print("press 's' to scan more document")
            print("press 'q' to quit")
            continue
        elif k1 == ord('u'):
            cv2.destroyWindow('Scanned photo')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 1)
            cv2.imwrite("E://pdf//scanned%d.jpg" % i, new)
            i = i + 1
            print("press 's' to scan more document")
            print("press 'q' to quit")
            continue
        elif k1 == ord('b'):
            cv2.destroyWindow('Scanned photo')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("E://pdf///scanned%d.jpg" % i, gray)
            i = i + 1
            print("press 's' to scan more document")
            print("press 'q' to quit")
            continue
        elif k1 == ord('e'):
            cv2.destroyWindow('Scanned photo')
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            cv2.imwrite("E://pdf///scanned%d.jpg" % i, hsv)
            i = i + 1
            lower_red = np.array([30, 150, 50])
            upper_red = np.array([255, 255, 180])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            res=cv2.bitwise_and(frame, frame, mask = mask)
            laplacian = cv2.Laplacian(frame,cv2.CV_64F)
            sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
            edges=cv2.Canny(frame, 100, 200)
            cv2.imwrite("E://pdf///scanned%d.jpg" % i, mask)
            i = i + 1
            cv2.imwrite("E://pdf///scanned%d.jpg" % i, laplacian)
            i = i + 1
            cv2.imwrite("E://pdf///scanned%d.jpg" % i, sobelx)
            i = i + 1
            cv2.imwrite("E://pdf///scanned%d.jpg" % i, sobely)
            i = i + 1
            cv2.imwrite("E://pdf///scanned%d.jpg" % i, edges)
            i = i + 1
            print("press 's' to scan more document")
            print("press 'q' to quit")
            continue
    elif k == ord('q'):
        ret = False
        break
cv2.destroyAllWindows()
imagelist = os.listdir("E://pdf")
pdf = FPDF()
for image in imagelist:
    image = "E://pdf//" + image
    pdf.add_page()
    pdf.image(image)
pdf.output("E://note.pdf", "F")
