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

        yardlines = yardlines[yardlines[:,0].argsort()]

        print(yardlines)

        try:
            hashmarks = LD.findHashmarks(img, yardlines, False)
        except:
            print("didn't find hashmarks")
            pass

        hashmarks = hashmarks[hashmarks[:,0].argsort()]

        intersects = LD.findIntersect(yardlines, hashmarks)

        print(intersects)

        transform = LD.findTransformMatrix(intersects)

        print(transform)

        """
        for pt in intersects:
            point = np.array([[pt[0]], [pt[1]], [1]])
            print(transform @ point)
            print()
        """



if __name__ == '__main__':
    main()