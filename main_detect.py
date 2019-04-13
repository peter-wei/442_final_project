import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main():
    filename = 'data/sample.png'

    img = cv.imread(filename)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    edges = cv.Canny(gray, 50, 150, apertureSize = 3)

    lines = cv.HoughLines(edges, 1, np.pi/180, 200)

    print(lines.shape)

    for line in lines:
        rho,theta = line[0]
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



if __name__ == '__main__':
    main()