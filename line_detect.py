import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


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

        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    plt.imshow(img)
    plt.show()

def checkMatch(newLine, merged):
    for i, line in enumerate(merged):
        if abs(line[0] - newLine[0]) < 30 and abs(line[1] - newLine[1]) < 5 * np.pi / 180:
            return i

    return -1

# Input: img is rgb image
def findYardlines(img):
    # convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    # find edge features
    edges = cv.Canny(gray, 50, 150, apertureSize = 3)

    # list of detected lines
    lines = cv.HoughLines(edges, 1, np.pi/180, 200)

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

    overlayLines(img, merged)

    return merged

def findHashmarks(img):
    # convert to lab colorscheme

    lab = cv.cvtColor(img, cv.COLOR_RGB2LAB)

    print(lab.shape)

    plt.imshow(lab[:,:,0])
    plt.show()