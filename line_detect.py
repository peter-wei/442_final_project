import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from numpy import linalg as LA


def findTransformMatrix(points):
    print(points)

    m = np.zeros((8, 9))

    count = 0

    for i in range(len(points)):
        if i < 2 or i > len(points) - 3:
            x1 = points[i][0]
            y1 = points[i][1]
            z1 = 1

            x2 = int(i/2)*5
            y2 = -(i%2 * 19/6)
            z2 = 1
            first = np.array([x1, y1, z1, 0, 0, 0, x2*x1, -x2*y1, -x2*z1])
            second = np.array([0, 0, 0, x1, y1, z1, -y2*x1, -y2*y1, -y2*z1])

            m[7-count*2] = first
            m[7-count*2 - 1] = second

            count += 1

    mtm = np.matmul(m.T, m)


    w, v = LA.eig(mtm)

    idx_min = np.argmin(w)

    answer = v[:, idx_min]

    answer = answer.reshape((3, 3))

    answer /= answer[2,2]

    return answer
    """
    A = np.zeros((len(points), 3))
    B = np.zeros((len(points), 3))

    for i in range(len(points)):
        

    return (np.linalg.inv(A.T @ A) @ A.T @ B).T
    """


def findIntersect(yardLines, hashmarks):
    points = []

    for line in yardLines:
        rho1, theta1 = line

        a1 = np.cos(theta1)
        b1 = np.sin(theta1)

        x01 = a1*rho1
        y01 = b1*rho1
        x1 = int(x01 + 2000*(-b1))
        y1 = int(y01 + 2000*(a1))
        x2 = int(x01 - 2000*(-b1))
        y2 = int(y01 - 2000*(a1))

        for mark in hashmarks:
            rho2, theta2 = mark

            a2 = np.cos(theta2)
            b2 = np.sin(theta2)

            x02 = a2*rho2
            y02 = b2*rho2
            x3 = int(x02 + 2000*(-b2))
            y3 = int(y02 + 2000*(a2))
            x4 = int(x02 - 2000*(-b2))
            y4 = int(y02 - 2000*(a2))

            t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))

            pt = []

            pt.append(x1 + t*(x2-x1))
            pt.append(y1 + t*(y2-y1))

            points.append(pt)

    return points

def overlayLines(img, lines):
    for line in lines:
        rho, theta = line

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))

        cv.line(img,(x1,y1),(x2,y2),(255,0,0),4)

    plt.imshow(img)
    plt.show()

def checkMatch(newLine, merged, drho=60, dtheta=5):
    for i, line in enumerate(merged):
        if abs(line[0] - newLine[0]) < drho and abs(line[1] - newLine[1]) < dtheta * np.pi / 180:
            return i

    return -1

# Input: img is rgb image
def findYardlines(img, makeplot=False):
    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # find edge features
    edges = cv.Canny(gray, 50, 150, apertureSize = 3)

    # list of detected lines
    lines = cv.HoughLines(edges, 2, np.pi/180, 400)

    # Fix theta, if theta > pi, subtract 2pi
    for i, line in enumerate(lines):
        rho, theta = line[0]

        if theta > np.pi / 2:
            lines[i, 0] = np.array([rho, theta - np.pi])

    # merge detected lines that are on the same yard line
    merged = []

    merged.append(lines[0, 0])

    for line in lines:
        # check all lines in merge to see if match
        match_idx = checkMatch(line[0], merged)

        if match_idx == -1:
            merged.append(line[0])
        else:
            merged[match_idx][0] = (merged[match_idx][0] + line[0][0]) / 2
            merged[match_idx][1] = (merged[match_idx][1] + line[0][1]) / 2

    merged = np.array(merged)

    # Unfix theta, so it matches original format
    for i, line in enumerate(merged):
        rho, theta = line

        if theta < 0:
            merged[i] = np.array([rho, theta + np.pi])

    print('Found ', merged.shape[0], ' yard lines in image')

    if makeplot:
        overlayLines(img, merged)

    return merged

def maskHash(yardLines, img):
    img_h = img.shape[0]

    yl_topx = []
    yl_botx = []

    for line in yardLines:
        rho, theta = line

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + y0 * b / a)

        x2 = int(x0 + (img_h - y0) / a * (-b))

        yl_topx.append(x1)
        yl_botx.append(x2)

    yl_topx.sort()
    yl_botx.sort()

    hash_topx = []
    hash_botx = []

    for i in range(len(yl_topx) - 1):
        top_val = np.arange(yl_topx[i], yl_topx[i+1], (yl_topx[i+1]-yl_topx[i])/5).astype(int)
        bot_val = np.arange(yl_botx[i], yl_botx[i+1], (yl_botx[i+1]-yl_botx[i])/5).astype(int)
        
        for j in range(1, 5):
            hash_topx.append(top_val[j])
            hash_botx.append(bot_val[j])

    mask = np.zeros((img.shape[0], img.shape[1]))

    for i in range(len(hash_topx)):
        cv.line(mask,(hash_topx[i],0),(hash_botx[i],img_h), 1, 16)

    return mask


def makeDoG(sigx, sigy):
    g1x = np.zeros((1, 101))
    g1x[0, 50] = 1
    g1y = np.zeros((101, 1))
    g1y[50, 0] = 1

    g2x = np.zeros((1, 101))
    g2x[0, 50] = 1
    g2y = np.zeros((101, 1))
    g2y[50, 0] = 1

    g1 = np.zeros((51, 51))
    g2 = np.zeros((51, 51))

    g1[25,25] = 1
    g2[25,25] = 1

    g1x = ndi.filters.gaussian_filter(g1x, sigma=sigx)
    g1y = ndi.filters.gaussian_filter(g1y, sigma=sigy)

    g1 = cv.filter2D(g1, -1, g1y)
    g1 = cv.filter2D(g1, -1, g1x)

    g2x = ndi.filters.gaussian_filter(g2x, sigma=1.6*sigx)
    g2y = ndi.filters.gaussian_filter(g2y, sigma=1.6*sigy)

    g2 = cv.filter2D(g2, -1, g2y)
    g2 = cv.filter2D(g2, -1, g2x)


    dog = g2 - g1

    return dog

def findHashmarks(img, yardLines, makeplot=False):
    print('START finding hashmarks')

    # Find hashmarks using blob detector
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    hue = hsv[:,:,0]
    sat = hsv[:,:,1]

    if makeplot:
        plt.imshow(hue)
        plt.show()

    if makeplot:
        plt.imshow(sat)
        plt.show()

    green = np.copy(img[:,:,1])

    hsv_mask = (hue > 38) * (hue < 55) * (sat > 85)

    white = np.ones(img.shape[0:2]) * 255

    white = white * hsv_mask

    if makeplot:
        plt.imshow(white)
        plt.show()

    # Create DoG (difference of gradients filter)
    dog = makeDoG(4, 8)

    white_dog = cv.filter2D(white,-1,dog)

    if makeplot:
        plt.imshow(white_dog)
        plt.show()

    white_dog = (white_dog > 40)

    if makeplot:
        plt.imshow(white_dog)
        plt.show()

    # Generate hashmark mask based on yardLines
    mask = maskHash(yardLines, img)

    dog_masked = (white_dog * mask).astype(np.uint8)

    if makeplot:
        plt.imshow(dog_masked)
        plt.show()

    # list of detected lines
    lines = cv.HoughLines(dog_masked, 15, np.pi/180, 1200)

    print(lines.shape[0], 'lines found')

    # merge detected lines that are on the same yard line
    merged = []

    merged.append(lines[0, 0])

    for line in lines:
        # check all lines in merge to see if match
        match_idx = checkMatch(line[0], merged, 100, 5)

        
        if match_idx == -1:
            merged.append(line[0])

    merged = np.array(merged)

    print('merged to', merged.shape[0], 'lines')

    #if makeplot:
    overlayLines(img, merged)

    print('END finding hashmarks')

    return merged