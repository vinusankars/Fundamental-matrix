
# coding: utf-8
import cv2 as cv
import numpy as np
from copy import copy
from matplotlib import pyplot as plt

#Reference: https://docs.opencv.org/3.4/da/de9/tutorial_py_epipolar_geometry.html

#Input image
# i1 = cv.resize(cv.imread("im0.png"), (100, 200))
# i2 = cv.resize(cv.imread("im1.png"), (100, 200))
i1 = cv.imread("im0.png")
i2 = cv.imread("im1.png")

#SIFT
sift = cv.xfeatures2d.SIFT_create()

k1, d1 = sift.detectAndCompute(i1, None)
k2, d2 = sift.detectAndCompute(i2, None)


#Mactch KP
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(d1,d2,k=2)


#Finding points for getting F matrix
good, p1, p2 = [], [], []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        p2.append(k2[m.trainIdx].pt)
        p1.append(k1[m.queryIdx].pt)
        
p1 = np.int32(p1)
p2 = np.int32(p2)

#Get F matrix
F, mask = cv.findFundamentalMat(p1,p2)
print("Fundamental matrix = \n", F)

p1 = p1[mask.ravel()==1]
p2 = p2[mask.ravel()==1]


#Calculate epipole lines
coords = []
pts = []

for i in range(i1.shape[0]):
    for j in range(i1.shape[1]):
        coords.append(cv.KeyPoint(j, i, 8))
        pts.append([j, i])

pts = np.array(pts, dtype = 'int32')
lines = cv.computeCorrespondEpilines(pts.reshape(-1, 1, 2), 1, F)
lines = lines.reshape(-1, 3)

#Get descriptors for pixels
_, des1 = sift.compute(i1, coords)
_, des2 = sift.compute(i2, coords)
des2 = des2.reshape(i2.shape[0], i2.shape[1], 128)


#Transform
img1 = np.zeros(i2.shape)

for i, p in enumerate(pts):
    if i%10000==0:
        print(i)
    a, b, c = lines[i]
    minNorm = np.inf
    X, Y = 0, 0
    for x in range(img1.shape[1]):
        y = int(-1*(a*x + c)/b)
        if -1 < y < img1.shape[0]:
            norm = np.linalg.norm(des2[y, x] - des1[i])
            if norm < minNorm:
                minNorm = norm
                X = x
                Y = y
    img1[p[1], p[0]] = i2[Y, X]


#Show output
img1 = img1.astype('uint8')
plt.figure(figsize=(5,5))
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.show()

#Save outputs
cv.imwrite('l.png', i1)
cv.imwrite('r.png', i2)
cv.imwrite('a.png', img1)