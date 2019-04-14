import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import line_detect as LD


def main():
    filename = 'data/sample.png'

    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    yardlines = LD.findYardlines(img)

    hashmarks = LD.findHashmarks(img, yardlines)



if __name__ == '__main__':
    main()