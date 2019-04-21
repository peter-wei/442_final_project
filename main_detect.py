import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import line_detect as LD


def main():
    for i in range(59):
        filename = 'data/sample_'

        if i < 10:
            filename += '0'

        filename += str(i) + '.jpg'

        print(filename)

        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        yardlines = LD.findYardlines(img, True)

        try:
            hashmarks = LD.findHashmarks(img, yardlines, False)
        except:
            print("didn't find hashmarks")
            pass



if __name__ == '__main__':
    main()