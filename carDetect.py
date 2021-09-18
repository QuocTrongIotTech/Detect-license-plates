import cv2
import imutils
import numpy as np

#load image
img = cv2.imread('GreenParking/0000_08244_b.jpg', cv2.IMREAD_COLOR)

#resize image
img = cv2.resize(img, dsize=(472, 303))

#Edge Detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#chuyen anh ve anh xam
Blur = cv2.bilateralFilter(gray, 11, 17, 17)#lam mo anh de giam nhieu
edge = cv2.Canny(Blur, 30, 200)#tim canh trong hinh

#tim contour trong anh canh
cnts = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#tim contour trong buc anh
cnts = imutils.grab_contours(cnts)#lay contour trong buc anh
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)#sap xep duong vien theo thu tu
#phan tich contour
for c in cnts:
    D = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05*D, True)
    if len(approx) == 4 and 6000 > cv2.contourArea(c) > 2000:
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
        break
mask = np.zeros(gray.shape, np.uint8)#tao ra mot buc anh den voi kich thuoc nhu anh gray
new_image = cv2.drawContours(mask, [approx], -1, 255, 2)#ve lai contour la bien so
new_image = cv2.bitwise_and(img, img, mask=mask)#rang buoc anh mask va anh goc
# Now crop
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cut = gray[topx:bottomx + 1, topy:bottomy + 1]
cv2.imshow('Anh bien so', Cut)
#Show image
#print(img.shape)
cv2.imshow('Anh goc', img)
#cv2.imshow('Anh xam', gray)
#cv2.imshow('Anh mo', Blur)
#cv2.imshow('Anh edge', edge)
cv2.waitKey()