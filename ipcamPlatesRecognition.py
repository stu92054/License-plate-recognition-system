import cv2
import subprocess
import numpy as np
import glob
import imutils
import threading
import pytesseract
import time
# from numba import njit

rtsp = "<your rtsp path>"
rtmp = '<your rtmp path>'


class MyThread(threading.Thread):
    # class:Thread
    def __init__(self, target=None, args=(), **kwargs):
        super(MyThread, self).__init__()
        self._target = target
        self._args = args
        self._kwargs = kwargs

    def run(self):
        if self._target is None:
            return
        self.__result__ = self._target(*self._args, **self._kwargs)

    def get_result(self):
        self.join()  # 當需要取得結果值的時候阻塞等待子執行緒完成
        return self.__result__


# 接收攝影機串流影像，採用多執行緒的方式，降低緩衝區堆疊圖幀的問題。
class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False

        # 攝影機連接。
        self.capture = cv2.VideoCapture(URL)

    def start(self):
        # 把程式放進子執行緒，daemon=True 表示該執行緒會隨著主執行緒關閉而關閉。
        print('ipcam started!')
        threading.Thread(target=self.queryframe, daemon=True, args=()).start()

    def stop(self):
        # 記得要設計停止無限迴圈的開關。
        self.isstop = True
        print('ipcam stopped!')

    def getframe(self):
        # 當有需要影像時，再回傳最新的影像。
        return self.Frame

    def getstatus(self):
        return self.status

    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()

        self.capture.release()


def adaptiveThreshold(plates):
    #for i, plate in enumerate(plates):
    img = plates
    #img = cv2.imread(plate)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    threshGauss = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51,
        27)

    final = resize(threshGauss)
    #cv2.imwrite("processed/plate.png", gray)
    return final


def resize(processed):
    #for i, image in enumerate(processed):
    #image = cv2.imread(image)
    image = processed

    ratio = 200.0 / image.shape[1]
    dim = (200, int(image.shape[0] * ratio))

    resizedCubic = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

    #cv2.imwrite("resized/plate{}.png".format(i), resizedCubic)
    final = addBorder(resizedCubic)
    return final


def addBorder(resized):
    #for i, image in enumerate(resized):
    #image = cv2.imread(image)
    image = resized

    bordersize = 10
    border = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize,
                                left=bordersize, right=bordersize,
                                borderType=cv2.BORDER_CONSTANT,
                                value=[255, 255, 255])

    #cv2.imwrite("borders/plate{}.png".format(i), border)
    final =  cleanOCR(border)
    return final


def cleanOCR(borders):
    detectedOCR = []

    #for i, image in enumerate(borders):
    #image = cv2.imread(image)
    image = borders

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

    cv2.imwrite("final/plate.png", gray)
    # OCR
    # Change PSM modes to 6
    config = '-l eng --oem 1 --psm 6'
    text = pytesseract.image_to_string(gray, config=config)

    validChars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                  'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                  'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6',
                  '7', '8', '9']

    cleanText = []

    for char in text:
        if char in validChars:
            cleanText.append(char)

    plate = ''.join(cleanText)

    detectedOCR.append(plate)

    return detectedOCR


def recognition(frame):
    start=time.time()
    # 定義recognition()函數辨識車牌
    # 使用到platesList、NumberPlateCnt兩個全域變數
    global platesList
    global NumberPlateCnt

    image = frame
    imagere = imutils.resize(frame, height=200)

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(imagere, cv2.COLOR_BGR2GRAY)


    max_ = float(gray.max())
    min_ = float(gray.min())
    stretchedimg = gray
    cv2.convertScaleAbs(gray, stretchedimg, (255 / (max_ - min_)), (255 * min_) / (max_ - min_))
    #stretchedimg = image.convertTo(gray,-1,(255 / (max_ - min_)),(255 * min_) / (max_ - min_))

    # 先定義一個元素結構
    r = 16
    h = w = r * 2 + 1
    kernel = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(kernel, (r, r), r, 1, -1)
    # 開運算
    openingimg = cv2.morphologyEx(stretchedimg, cv2.MORPH_OPEN, kernel)

    # 獲取差分圖
    strtimg = cv2.absdiff(stretchedimg, openingimg)

    # 在對圖像進行邊緣檢測之前，，先對圖像進行二值化
    ret, binary_img = cv2.threshold(strtimg, 125, 255, cv2.THRESH_BINARY)

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

    # Find Edges of the grayscale image
    edged = cv2.Canny(kernel_dilated, 50, 300)
    edged = kernel_dilated

    # Find contours based on Edges
    (cnts, _) = cv2.findContours(edged.copy(),
                                 cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours based on their area keeping minimum required area as '30'
    # (anything smaller than this will not be considered)把上面Contours算出來的面積排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    if lock.acquire():
        try:
            NumberPlateCnt = None  # we currently have no Number plate contour

            # loop over our contours to find
            # the best possible approximate contour of number plate
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                if aspect_ratio < 5 and aspect_ratio > 2.5:
                    if len(approx) == 4:  # Select the contour with 4 corners車牌四方形
                        NumberPlateCnt = approx
                        # This is our approx Number Plate Contour
                        break
                    else:
                        NumberPlateCnt = np.array(
                            [[[x, y]], [[x + w, y]],
                             [[x + w, y + h]], [[x, y + h]]])

            # Drawing the selected contour on the original image
            # pdb.set_trace()
            if NumberPlateCnt is None:
                NumberPlateCnt = np.array([[[0, 1]], [[1, 0]],
                                           [[1, 1]], [[0, 1]]])
        finally:
            lock.release()
    # ans=image[test[1]:test[1]+(test2[1]-test[1]),test[0]:test[0]+(test1[0]-test[0])]
    roi = image.copy()
    rows, cols, ch = roi.shape
    roi[0:0 + rows, 0:0 + cols] = [0, 0, 0]

    cv2.fillConvexPoly(roi, NumberPlateCnt, (255, 255, 255))

    Xs = [i[0][0] for i in NumberPlateCnt]
    Ys = [i[0][1] for i in NumberPlateCnt]
    x1 = int(min(Xs)/200*frame.shape[0])
    x2 = int(max(Xs)/200*frame.shape[0])
    y1 = int(min(Ys)/200*frame.shape[0])
    y2 = int(max(Ys)/200*frame.shape[0])
    hight = y2 - y1
    width = x2 - x1
    #print(width/hight)
    cropImg = image[y1:y1 + hight, x1:x1 + width]

    cv2.imwrite("plates/final.png", cropImg)

    #plates = glob.glob("plates/*.png")
    platesList = adaptiveThreshold(cropImg)
    #processed = glob.glob("processed/*.png")
    #resize(processed)
    #resized = glob.glob("resized/*.png")
    #addBorder(resized)
    #bordered = glob.glob("borders/*.png")
    #platesList = cleanOCR(bordered)
    end = time.time()
    print(end-start)
    return platesList


# 連接攝影機
ipcam = ipcamCapture(rtsp)
# 啟動子執行緒
ipcam.start()
# 暫停1秒，確保影像已經填充
time.sleep(1)
# 读取视频并获取属性
cap = cv2.VideoCapture(rtsp)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
sizeStr = str(size[0]) + 'x' + str(size[1])

command = ['ffmpeg',
           '-y', '-an',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-s', sizeStr,
           '-r', '30',
           '-i', '-',
           '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p',
           '-preset', 'ultrafast',
           '-f', 'flv',
           '-bufsize', '6000k',
           rtmp]

pipe = subprocess.Popen(command
                        , stdin=subprocess.PIPE
                        )

t = MyThread(target=recognition, args=())
lock = threading.Lock()
platesList = ['']
NumberPlateCnt = np.array([[[0, 1]], [[1, 0]], [[1, 1]], [[0, 1]]])

while(True):
    # success, frame = cap.read()
    # 使用 getframe 取得最新的影像
    frame = ipcam.getframe()
    if ipcam.getstatus():
        if lock.acquire():
            try:
                # 將resize的座標resize
                OriginalNumberPlateCnt = np.array([[[NumberPlateCnt[0][0][0] / 200
                                                     * frame.shape[0],
                                                     NumberPlateCnt[0][0][1] / (
                                                     frame.shape[1] * 200 /
                                                     frame.shape[0])
                                                     * frame.shape[1]]],
                                                   [[NumberPlateCnt[1][0][0] / 200
                                                    * frame.shape[0],
                                                     NumberPlateCnt[1][0][1] / (
                                                     frame.shape[1] * 200 /
                                                     frame.shape[0])
                                                     * frame.shape[1]]],
                                                   [[NumberPlateCnt[2][0][0] / 200
                                                    * frame.shape[0],
                                                     NumberPlateCnt[2][0][1] / (
                                                     frame.shape[1] * 200 /
                                                     frame.shape[0])
                                                     * frame.shape[1]]],
                                                   [[NumberPlateCnt[3][0][0] / 200
                                                    * frame.shape[0],
                                                     NumberPlateCnt[3][0][1] / (
                                                     frame.shape[1] * 200 /
                                                     frame.shape[0])
                                                     * frame.shape[1]]]])
            finally:
                lock.release()

        cv2.drawContours(frame, [OriginalNumberPlateCnt.astype(int)], -1,
                         (0, 255, 0), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            cv2.putText(frame, text=platesList[0], org=(
                100, 250), fontFace=font, fontScale=2, color=(0, 0, 255), thickness=3)
        except IndexError:
            pass
        # cv2.putText(frame, text=platesList[0], org=(
        #    100, 250), fontFace=font, fontScale=1, color=(0, 0, 255), thickness=1)

        # cv2.imshow('frame',frame)
        pipe.stdin.write(frame.tostring())

        # 如果thread不在執行，則執行判斷車牌的thread
        if t.is_alive() is False:
            t = MyThread(target=recognition, args=(frame,))
            t.start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
pipe.terminate()
