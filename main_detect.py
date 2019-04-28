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

        try:
            hashmarks = LD.findHashmarks(img, yardlines, False)
        except:
            print("didn't find hashmarks")
            pass

        hashmarks = hashmarks[hashmarks[:,0].argsort()]

        intersects = LD.findIntersect(yardlines, hashmarks)

        transform = LD.findTransformMatrix(intersects)

        print(transform)

        """
        for pt in intersects:
            point = np.array([[pt[0]], [pt[1]], [1]])
            print(transform @ point)
            print()
        """

        location_file = 'player_locations/sample_'

        if i < 10:
            location_file += '0'

        location_file += str(i) + '.txt'

        file1 = open(location_file, 'r')

        coords = file1.read()

        player_points = []

        for point in coords.splitlines():
            tmp_coords = point.split(' ')

            player_points.append([float(tmp_coords[0]), float(tmp_coords[1]), 1])

        player_points = np.array(player_points)

        transformed_points = np.zeros((player_points.shape[0], 2))

        print(player_points)

        for i, pt in enumerate(player_points):
            point = np.array([[pt[1]], [pt[0]], [1]])

            tmp = transform @ point;

            transformed_points[i] = np.array([tmp[0,0]/tmp[2,0], tmp[1,0]/tmp[2,0]])


        intersects_plot = np.zeros((len(intersects), 2))

        for i, pt in enumerate(intersects):
            point = np.array([[pt[0]], [pt[1]], [1]])
            tmp = transform @ point

            intersects_plot[i] = np.array([tmp[0,0]/tmp[2,0], tmp[1,0]/tmp[2,0]])

        print(intersects_plot)

        plt.scatter(transformed_points[:,0], transformed_points[:,1])
        plt.scatter(intersects_plot[:,0], intersects_plot[:,1], c='red')
        plt.show()


if __name__ == '__main__':
    main()