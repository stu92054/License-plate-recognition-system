# coding=utf-8
import cv2
import numpy as np
import glob
import pytesseract
import imutils
import os.path as path
# pip insyall pytesseract
import pdb


def adaptiveThreshold(plates):
    for i, plate in enumerate(plates):
        img = cv2.imread(plate)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow('gray', gray)

        threshGauss = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 27)
        threshMean = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
        cv2.imshow('threshMean', threshGauss)
        cv2.imwrite("processed/plate{}.png".format(i), threshGauss)


def resize(processed):
    for i, image in enumerate(processed):
        image = cv2.imread(image)

        ratio = 200.0 / image.shape[1]
        dim = (200, int(image.shape[0] * ratio))

        resizedCubic = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite("resized/plate{}.png".format(i), resizedCubic)


def addBorder(resized):
    for i, image in enumerate(resized):
        image = cv2.imread(image)

        bordersize = 10
        border = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                    borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

        cv2.imwrite("borders/plate{}.png".format(i), border)


def cleanOCR(borders):
    detectedOCR = []

    for i, image in enumerate(borders):
        image = cv2.imread(image)

        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                                minLineLength=100, maxLineGap=80)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        gray = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)
        cv2.imwrite("final/plate{}.png".format(i), gray)

        # OCR
        config = '-l eng --oem 1 --psm 6'
        text = pytesseract.image_to_string(gray, config=config)
        print(text)

        validChars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        cleanText = []

        for char in text:
            if char in validChars:
                cleanText.append(char)

        plate = ''.join(cleanText)
        print(plate)

        detectedOCR.append(plate)

        cv2.imshow('img', gray)

    return detectedOCR


# Read the image file
file_paths = glob.glob(path.join("car/*.jpg"))
for path3 in file_paths:
    # image = cv2.imread('cars/car4.png')
    image = cv2.imread(path3)
    # Resize the image - change width to 500
    image = imutils.resize(image, width=500)

    # Display the original image
    cv2.imshow("Original Image", image)

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("1 - Grayscale Conversion", gray)

    # 像素拉伸
    def stretch(img):
        max_ = float(img.max())
        min_ = float(img.min())

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j] = (255 / (max_ - min_)) * img[i, j] - \
                    (255 * min_) / (max_ - min_)
        return img
    max_ = float(gray.max())
    min_ = float(gray.min())
    stretchedimg = gray
    cv2.convertScaleAbs(gray, stretchedimg, (255 / (max_ - min_)), (255 * min_) / (max_ - min_))
    cv2.imshow('2 - stretchedimg', stretchedimg)

    # 先定義一個元素結構
    r = 16
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(kernel, (r, r), r, 1, -1)
    # 開運算
    openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)
    cv2.imshow('3 - openingimg', openingimg)

    # 獲取差分圖
    strtimg = cv2.absdiff(stretchedimg, openingimg)
    cv2.imshow('4 - strtimg', strtimg)

    # 在對圖像進行邊緣檢測之前，，先對圖像進行二值化
    ret, binary_img = cv2.threshold(strtimg, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow('5 - binary_img', binary_img)

    # 進行閉運算
    kernel = np.ones((5, 19), np.uint8)
    closing_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # 進行開運算
    opening_img = cv2.morphologyEx(closing_img, cv2.MORPH_OPEN, kernel)
    # 再次進行開運算
    kernel = np.ones((11, 5), np.uint8)
    opening_img = cv2.morphologyEx(opening_img, cv2.MORPH_OPEN, kernel)
    # 膨脹
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_dilated = cv2.dilate(opening_img, kernel_2)
    cv2.imshow('7 - kernel_dilated', kernel_dilated)

    # Find Edges of the grayscale image
    edged = cv2.Canny(kernel_dilated, 50, 300)
    cv2.imshow("8 - Canny Edges", edged)
    edged = kernel_dilated

    # Find contours based on Edges
    (cnts, _) = cv2.findContours(edged.copy(),
                                 cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)把上面Contours算出來的面積排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None  # we currently have no Number plate contour

    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        if aspect_ratio < 5.5 and aspect_ratio > 1.5:
            if len(approx) == 4:  # Select the contour with 4 corners車牌四方形
                NumberPlateCnt = approx  # This is our approx Number Plate Contour
                break
            else:
                NumberPlateCnt = np.array(
                    [[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]])

    # Drawing the selected contour on the original image
    drawcon = image.copy()
    # pdb.set_trace()
    print(NumberPlateCnt)
    cv2.drawContours(drawcon, [NumberPlateCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Final Image With Number Plate Detected", drawcon)
    print(NumberPlateCnt[0][0], NumberPlateCnt[1],
          NumberPlateCnt[2], NumberPlateCnt[3])
    test = NumberPlateCnt[0][0]
    test1 = NumberPlateCnt[1][0]
    test2 = NumberPlateCnt[2][0]
    print(test[0], test[1], test1[0], test2[1])
    roi = image.copy()
    rows, cols, ch = roi.shape
    roi[0:0 + rows, 0:0 + cols] = [0, 0, 0]

    cv2.fillConvexPoly(roi, NumberPlateCnt, (255, 255, 255))

    final = cv2.bitwise_and(image, roi)
    Xs = [i[0][0] for i in NumberPlateCnt]
    Ys = [i[0][1] for i in NumberPlateCnt]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = image[y1:y1 + hight, x1:x1 + width]

    cv2.imshow("black", cropImg)
    cv2.imwrite("plates/final.png", cropImg)
    # cv2.imwrite("plates/test.png",ans)
    #cv2.imshow("Number Plate Detected", ans)

    plates = glob.glob("plates/*.png")
    adaptiveThreshold(plates)
    processed = glob.glob("processed/*.png")
    resize(processed)
    resized = glob.glob("resized/*.png")
    addBorder(resized)
    bordered = glob.glob("borders/*.png")
    platesList = cleanOCR(bordered)

    print(platesList)
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        cv2.putText(drawcon, text=platesList[0], org=(
            100, 250), fontFace=font, fontScale=1, color=(0, 0, 255), thickness=2)
    except IndexError:
        pass
    cv2.imshow("fianl", drawcon)
    cv2.waitKey(0)
