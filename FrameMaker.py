import cv2
import math
import numpy as np
import mediapipe as mp
import pygame
import time

class FrameMaker:
    def __init__(self):
        self.inputFrame = None
        self.outputFrame = None
        self.cannyImage = None

        self.printGuitarEdge = False
        self.printHandLandmark = False
        self.printHorizontalLines = False
        self.printFilteredDots = False
        self.printStringPositions = False

        self.codeMode = False
        self.strokeMode = False
        self.songMode = False
        self.fakePlayMode = False

        self.yCutOff = 300

        self.stringOffset = [1.041666667,
                            1.04,
                            1.076923077,
                            1.071428571,
                            1.066666667,
                            1.0625,
                            1.058823529,
                            1.055555556,
                            1.052631579,
                            1.05,
                            1.071428571,
                            1.066666667,
                            1.041666667,
                            1.08,
                            1.037037037,
                            1.071428571,
                            1.066666667,
                            1.0625,
                            1.088235294]
        self.filterStart = False
        self.stringPosition = []
        self.lastStringPosition = []

        self.frameWidth = 1280
        self.frameHeight = 720

        self.gray = (100, 100, 100)
        self.green = (0, 255, 0)
        self.yellow = (0, 255, 255)
        self.downArrow = np.array([[10, 0], [10, 50], [3, 50], [11, 70], [19, 50], [12, 50], [12, 0]], np.int32)
        self.upArrow = np.array([[0, 20], [7, 20], [7, 70], [9, 70], [9, 20], [16, 20], [8, 0]], np.int32)

        # Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.3)
        self.leftFingerPosition = []
        self.rightFingerPosition = [0, 0]
        self.isLeft = False
        self.isRight = False
        self.isRightFingerUp = None

        # Guitar Tracking
        self.cannyMin = 50;
        self.cannyMax = 150;
        self.houghThreshold = 190

        self.guitarLowerY1 = -9999
        self.guitarUpperY1 = 9999
        self.guitarLowerY2 = -9999
        self.guitarUpperY2 = 9999

        self.leftCutOff = 10
        self.rightCutOff = 890
        self.firstDotX = 0

        # Code
        self.codes = {'G': 0, 'D': 1, 'Em': 2, 'Bm': 3, 'C': 4, 'G7': 5, 'Am': 6}
        self.codePosition = [
            [[0, 2, 4], [1, 1, 5], [2, 1, 0]],
            [[0, 2, 2], [1, 2, 0], [2, 1, 1]],
            [[1, 2, 4], [2, 2, 3]],
            [[0, 2, 0], [0, 2, 4], [1, 1, 1], [3, 0, 3], [2, 0, 4]],
            [[0, 3, 1], [1, 2, 3], [2, 1, 4]],
            [[0, 3, 0], [1, 2, 4], [2, 1, 5]],
            [[0, 3, 1], [1, 2, 3], [2, 2, 2]]
        ]
        self.curCode = 'G'
        self.isFingerCodeMatch = [0, 0, 0, 0]
        self.stringRatio = [35 / 43, 21 / 43, 7 / 43, -7 / 43, -21 / 43, -35 / 43]
        self.codeCorrect = False

        # Sound
        pygame.mixer.init()
        self.gSound = pygame.mixer.Sound('./sound/g.mp3')
        self.dSound = pygame.mixer.Sound('./sound/d.mp3')
        self.emSound = pygame.mixer.Sound('./sound/em.mp3')
        self.bmSound = pygame.mixer.Sound('./sound/bm.mp3')
        self.cSound = pygame.mixer.Sound('./sound/c.mp3')
        self.g7Sound = pygame.mixer.Sound('./sound/g7.mp3')
        self.amSound = pygame.mixer.Sound('./sound/am.mp3')
        self.strokeSound = pygame.mixer.Sound('./sound/stroke.mp3')
        self.codeSounds = [self.gSound, self.dSound, self.emSound, self.bmSound, self.cSound, self.g7Sound, self.amSound]

        # Stroke
        self.startTime = None
        self.isStrokeStart = False
        self.isShowStroke = False
        self.isDoStroke = False
        self.strokeArr = [True, False, False, True, True, False, True, True]
        self.strokeIdx = 0

        # Song
        self.songCodes = ['G', 'D', 'Em', 'Bm', 'C', 'G', 'Am', 'D', 'G', 'D', 'Em', 'G7', 'C', 'G', 'Am', 'D']
        self.songCodeIdx = 0
        self.isDoSong = False

        # Image
        self.downArrow = cv2.imread('./icon/downArrow.png', 1)
        self.upArrow = cv2.imread('./icon/upArrow.png', 1)
        self.downArrowGray = cv2.imread('./icon/downArrow_gray.png', 1)
        self.upArrowGray = cv2.imread('./icon/upArrow_gray.png', 1)

        print("make")

    def setFrame(self, src):
        self.inputFrame = src
        self.outputFrame = src

    def getFrame(self):
        return self.outputFrame

    def getCode(self):
        return self.curCode, self.songCodeIdx

    def setFakePlayMode(self):
        self.fakePlayMode = not self.fakePlayMode

    def getFakePlayMode(self):
        return self.fakePlayMode

    def setCodeMode(self):
        self.codeMode = True
        self.strokeMode = False
        self.songMode = False

    def setStrokeMode(self):
        self.codeMode = False
        self.strokeMode = True
        self.songMode = False

    def setSongMode(self):
        self.codeMode = False
        self.strokeMode = False
        self.songMode = True

    def setDoSong(self):
        self.isDoSong = not self.isDoSong

    def doSong(self):
        if not self.isDoSong or self.songCodeIdx == len(self.songCodes):
            self.songCodeIdx = 0
            self.strokeIdx = 0
            self.isDoSong = False
            return

        self.curCode = self.songCodes[self.songCodeIdx]

        self.drawCode()

        if self.strokeIdx == 8:
            self.strokeIdx = 0
            self.songCodeIdx = self.songCodeIdx + 1

        img = None
        if self.strokeIdx % 2 == 0:
            if self.strokeArr[self.strokeIdx]:
                img = self.downArrow
            else:
                img = self.downArrowGray
        else:
            if self.strokeArr[self.strokeIdx]:
                img = self.upArrow
            else:
                img = self.upArrowGray

        rows, cols, channels = img.shape  # 로고파일 픽셀값 저장
        x = int(self.firstDotX) + 100
        y = int(self.guitarUpperY2 + self.yCutOff)
        roi = self.outputFrame[y:rows + y, x:cols + x]  # 로고파일 필셀값을 관심영역(ROI)으로 저장함.

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 로고파일의 색상을 그레이로 변경
        ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)  # 배경은 흰색으로, 그림을 검정색으로 변경
        mask_inv = cv2.bitwise_not(mask)

        src1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)  # 배경에서만 연산 = src1 배경 복사
        src2_fg = cv2.bitwise_and(img, img, mask=mask)  # 로고에서만 연산

        dst = cv2.bitwise_or(src1_bg, src2_fg)  # src1_bg와 src2_fg를 합성

        self.outputFrame[y:rows + y, x:cols + x] = dst  # src1에 dst값 합성

    def setShowStroke(self):
        self.isDoStroke = False
        self.isShowStroke = not self.isShowStroke

    def getShowStroke(self):
        return self.isShowStroke

    def getDoStroke(self):
        return self.isDoStroke

    def setDoStroke(self):
        self.isShowStroke = False
        self.isDoStroke = not self.isDoStroke

    def doStroke(self):
        if not self.isDoStroke or self.strokeIdx == 8:
            self.strokeIdx = 0
            return

        img = None
        if self.strokeIdx % 2 == 0:
            if self.strokeArr[self.strokeIdx]:
                img = self.downArrow
            else:
                img = self.downArrowGray
        else:
            if self.strokeArr[self.strokeIdx]:
                img = self.upArrow
            else:
                img = self.upArrowGray

        rows, cols, channels = img.shape # 로고파일 픽셀값 저장
        x = int(self.firstDotX) + 100
        y = int(self.guitarUpperY2 + self.yCutOff+20)
        roi = self.outputFrame[y:rows + y, x:cols + x]  # 로고파일 필셀값을 관심영역(ROI)으로 저장함.

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 로고파일의 색상을 그레이로 변경
        ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)  # 배경은 흰색으로, 그림을 검정색으로 변경
        mask_inv = cv2.bitwise_not(mask)

        src1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)  # 배경에서만 연산 = src1 배경 복사
        src2_fg = cv2.bitwise_and(img, img, mask=mask)  # 로고에서만 연산

        dst = cv2.bitwise_or(src1_bg, src2_fg)  # src1_bg와 src2_fg를 합성

        self.outputFrame[y:rows + y, x:cols + x] = dst  # src1에 dst값 합성

    def showStroke(self):
        sound = self.strokeSound

        if not self.isShowStroke:
            sound.stop()
            return

        if not self.isStrokeStart:
            sound.stop()
            sound.play()
            self.startTime = time.time()
            self.isStrokeStart = True

        img = self.downArrow

        i = 1
        if self.isStrokeStart:
            t = time.time()
            if t - self.startTime > 0.375 * i:
                i = i + 1
                img = self.upArrowGray

            if t - self.startTime > 0.375 * i:
                i = i + 1
                img = self.downArrowGray

            if t - self.startTime > 0.375 * i-0.01:
                i = i + 1
                img = self.upArrow

            if t - self.startTime > 0.375 * i:
                i = i + 1
                img = self.downArrow

            if t - self.startTime > 0.375 * i:
                i = i + 1
                img = self.upArrowGray

            if t - self.startTime > 0.375 * i:
                i = i + 1
                img = self.downArrow

            if t - self.startTime > 0.375 * i:
                i = i + 1
                img = self.upArrow

            if t - self.startTime > 0.375 * i:
                # self.isShowStroke = False
                self.isStrokeStart = False


        rows, cols, channels = img.shape  # 로고파일 픽셀값 저장
        x = int(self.firstDotX) + 100
        y = int(self.guitarUpperY2 + self.yCutOff)
        roi = self.outputFrame[y:rows + y, x:cols + x]  # 로고파일 필셀값을 관심영역(ROI)으로 저장함.

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 로고파일의 색상을 그레이로 변경
        ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)  # 배경은 흰색으로, 그림을 검정색으로 변경
        mask_inv = cv2.bitwise_not(mask)

        src1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)  # 배경에서만 연산 = src1 배경 복사
        src2_fg = cv2.bitwise_and(img, img, mask=mask)  # 로고에서만 연산

        dst = cv2.bitwise_or(src1_bg, src2_fg)  # src1_bg와 src2_fg를 합성

        self.outputFrame[y:rows + y, x:cols + x] = dst  # src1에 dst값 합성

    def setCode(self, code):
        self.curCode = code
        self.codeCorrect = False

    def codeJudge(self):
        codeIdx = self.codes[self.curCode]

        for i in self.codePosition[codeIdx]:
            output = False

            fingerIdx = i[0]

            fingerX = self.leftFingerPosition[fingerIdx][0]
            fingerY = self.leftFingerPosition[fingerIdx][1]
            fingerPos = (fingerX, fingerY)

            guitarUpper = (10, int(self.guitarUpperY1) + self.yCutOff - 10, 890, int(self.guitarUpperY2) + self.yCutOff - 10)
            guitarLower = (10, int(self.guitarLowerY1) + self.yCutOff + 10, 890, int(self.guitarLowerY2) + self.yCutOff + 10)

            isInGuitar = self.isInLines(fingerPos, guitarUpper, guitarLower)

            if isInGuitar == False:
                self.isFingerCodeMatch[fingerIdx] =  self.isFingerCodeMatch[fingerIdx] * 0.9
                if self.isFingerCodeMatch[fingerIdx] < 0.5:
                    cv2.line(self.outputFrame, fingerPos, fingerPos, (0, 0, 255), 15)
                    cv2.putText(self.outputFrame, str(fingerIdx + 1), (fingerX - 5, fingerY + 5), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 0, 0), 2)
                continue

            x = i[1]
            y = i[2]

            leftCodeLineX1 = self.stringPosition[x + 1][0][0]
            leftCodeLineY1 = self.stringPosition[x + 1][0][1] + self.yCutOff
            leftCodeLineX2 = self.stringPosition[x + 1][5][0]
            leftCodeLineY2 = self.stringPosition[x + 1][5][1] + self.yCutOff
            leftCodeLine = (leftCodeLineX1, leftCodeLineY1, leftCodeLineX2, leftCodeLineY2)

            rightCodeLineX1 = self.stringPosition[x][0][0]
            rightCodeLineY1 = self.stringPosition[x][0][1] + self.yCutOff
            rightCodeLineX2 = self.stringPosition[x][5][0]
            rightCodeLineY2 = self.stringPosition[x][5][1] + self.yCutOff
            rightCodeLine = (rightCodeLineX1, rightCodeLineY1, rightCodeLineX2, rightCodeLineY2)

            isInCodeLine = self.isInLines(fingerPos, leftCodeLine, rightCodeLine)

            if isInCodeLine == False:
                self.isFingerCodeMatch[fingerIdx] = self.isFingerCodeMatch[fingerIdx] * 0.9
                if self.isFingerCodeMatch[fingerIdx] < 0.5:
                    cv2.line(self.outputFrame, fingerPos, fingerPos, (0, 0, 255), 15)
                    cv2.putText(self.outputFrame, str(fingerIdx + 1), (fingerX - 5, fingerY + 5), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 0, 0),
                                2)
                continue

            guitarUpper = (10, int(self.guitarUpperY1) + self.yCutOff, 890, int(self.guitarUpperY2) + self.yCutOff)
            guitarLower = (10, int(self.guitarLowerY1) + self.yCutOff, 890, int(self.guitarLowerY2) + self.yCutOff)
            distGuitarUpper = self.getDistanceBtwLineDot(fingerPos, guitarUpper)
            distGuitarLower = self.getDistanceBtwLineDot(fingerPos, guitarLower)

            curStringRatio = (distGuitarUpper - distGuitarLower) / (distGuitarUpper + distGuitarLower)
            realStringRatio = self.stringRatio[y]

            if abs(realStringRatio - curStringRatio) > 10 / 43:
                self.isFingerCodeMatch[fingerIdx] = self.isFingerCodeMatch[fingerIdx] * 0.9
                if self.isFingerCodeMatch[fingerIdx] < 0.5:
                    cv2.line(self.outputFrame, fingerPos, fingerPos, (0, 0, 255), 15)
                    cv2.putText(self.outputFrame, str(fingerIdx + 1), (fingerX - 5, fingerY + 5), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 0, 0),
                                2)
                continue

            self.isFingerCodeMatch[fingerIdx] = self.isFingerCodeMatch[fingerIdx] * 0.9 + 0.1


    def drawCode(self):
        codeIdx = self.codes[self.curCode]

        isAllInGuitar = True

        for i in self.codePosition[codeIdx]:
            fingerIdx = i[0]
            x = i[1]
            y = i[2]

            codeX = (self.stringPosition[x][y][0] + self.stringPosition[x + 1][y][0]) / 2
            codeY = (self.stringPosition[x][y][1] + self.stringPosition[x + 1][y][1]) / 2 + 300

            guitarUpper = (10, int(self.guitarUpperY1) + self.yCutOff, 890, int(self.guitarUpperY2) + self.yCutOff)
            guitarLower = (10, int(self.guitarLowerY1) + self.yCutOff, 890, int(self.guitarLowerY2) + self.yCutOff)

            isInGuitar = self.isInLines((codeX, codeY), guitarUpper, guitarLower)

            if isInGuitar == False:
                isAllInGuitar = False
                break

            fingerIdx = fingerIdx + 1

        if isAllInGuitar:
            correct = True
            for i in self.codePosition[codeIdx]:
                fingerIdx = i[0]
                x = i[1]
                y = i[2]

                codeX = (self.stringPosition[x][y][0] + self.stringPosition[x + 1][y][0]) / 2
                codeY = (self.stringPosition[x][y][1] + self.stringPosition[x + 1][y][1]) / 2 + 300
                color = (0, 255, 255)

                if self.isFingerCodeMatch[fingerIdx] > 0.5:
                    color = (255, 0, 0)
                else:
                    correct = False

                cv2.line(self.outputFrame, (int(codeX), int(codeY)), (int(codeX), int(codeY)), color, 15)
                cv2.putText(self.outputFrame, str(fingerIdx + 1), (int(codeX) - 5, int(codeY) + 5), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 0), 2)
                fingerIdx = fingerIdx + 1

            if not self.codeCorrect and correct:
                sound = self.codeSounds[self.codes[self.curCode]]
                sound.stop()
                sound.play()
                self.codeCorrect = True

    def getDistanceBtwLineDot(self, dot, line):

        x0 = line[0]
        y0 = line[1]
        x1 = line[2]
        y1 = line[3]
        x2 = dot[0]
        y2 = dot[1]

        dist = abs((x0 - x1) * (y2 - y1) - (x2 - x1) * (y0 - y1))
        denomi = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) + 0.000001
        dist = dist / denomi

        return dist

    def isInLines(self, dot, line1, line2):
        output = None

        dotX = dot[0]
        dotY = dot[1]

        line1X1 = line1[0]
        line1Y1 = line1[1]
        line1X2 = line1[2]
        line1Y2 = line1[3]

        line2X1 = line2[0]
        line2Y1 = line2[1]
        line2X2 = line2[2]
        line2Y2 = line2[3]

        line1Vector1 = (line1X1 - dotX, line1Y1 - dotY)
        line1Vector2 = (line1X2 - dotX, line1Y2 - dotY)
        cross1 = line1Vector1[0] * line1Vector2[1] - line1Vector1[1] * line1Vector2[0]

        line2Vector1 = (line2X1 - dotX, line2Y1 - dotY)
        line2Vector2 = (line2X2 - dotX, line2Y2 - dotY)
        cross2 = line2Vector1[0] * line2Vector2[1] - line2Vector1[1] * line2Vector2[0]

        if cross1 * cross2 > 0:
            output = False
        else:
            output = True

        return output

    def getIntersection(self, rho1, theta1, rho2, theta2):
        a = np.cos(theta1)
        b = np.sin(theta1)
        x0 = a * rho1
        y0 = b * rho1
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        a = np.cos(theta2)
        b = np.sin(theta2)
        x0 = a * rho2
        y0 = b * rho2
        x3 = int(x0 + 1000 * (-b))
        y3 = int(y0 + 1000 * (a))
        x4 = int(x0 - 1000 * (-b))
        y4 = int(y0 - 1000 * (a))

        px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        denomi = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 0.000001
        px = px / denomi
        py = py / denomi

        return px, py

    def getDistance(self, x1, y1, x2, y2):
        x = x1 - x2
        y = y1 - y2

        d = x ** 2 + y ** 2
        d = math.sqrt(d)

        return d

    def getIntersectionDot(self, x1, y1, x2, y2, x3, y3, x4, y4):
        px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)
        py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)
        denomi = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) + 0.000001
        px = px / denomi
        py = py / denomi

        return px, py

    def canny(self):
        self.cannyImage = cv2.Canny(self.inputFrame, self.cannyMin, self.cannyMax, None, 3)

    def detectHand(self):
        image = cv2.cvtColor(self.inputFrame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)

        self.isLeft = False
        self.isRight = False

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                self.isLeft = False
                self.isRight = False
                for id, lm in enumerate(hand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)

                    if id == 0:
                        if cx < self.frameWidth / 2:
                            self.isLeft = True
                            self.leftFingerPosition = []
                        else:
                            self.isRight = True
                            self.rightFingerPosition = [0, 0]

                    if self.isLeft and (id == 8 or id == 12 or id == 16 or id == 20):
                        self.leftFingerPosition.append([cx, cy])

                    if self.isRight and id == 4:
                        self.rightFingerPosition = [int(cx), int(cy)]

                if self.printHandLandmark:
                    self.mp_drawing.draw_landmarks(self.outputFrame, hand, self.mp_hands.HAND_CONNECTIONS)

        vector1 = [10 - self.rightFingerPosition[0], self.guitarUpperY1 + self.yCutOff + 40 - self.rightFingerPosition[1]]
        vector2 = [890 - self.rightFingerPosition[0], self.guitarUpperY2 + self.yCutOff + 40 - self.rightFingerPosition[1]]

        crossProduct = vector1[0] * vector2[1] - vector1[1] * vector2[0]

        # cv2.line(self.outputFrame, (10, int(self.guitarUpperY1) + self.yCutOff + 40), (890, int(self.guitarUpperY2) + self.yCutOff + 40), (0, 0, 255),
        #          2)

        if crossProduct > 0:
            if self.isRightFingerUp == False:
                # cv2.line(self.outputFrame, (self.rightFingerPosition), (self.rightFingerPosition), (255, 255, 255), 50)
                if self.fakePlayMode:
                    if self.songMode and self.strokeArr[self.strokeIdx]:
                        sound = self.codeSounds[self.codes[self.curCode]]
                        sound.stop()
                        sound.play()

                    if self.codeMode:
                        sound = self.codeSounds[self.codes[self.curCode]]
                        sound.stop()
                        sound.play()

                if self.strokeMode or self.songMode:
                    self.strokeIdx = self.strokeIdx + 1

            self.isRightFingerUp = True
            # cv2.line(self.outputFrame, (self.rightFingerPosition), (self.rightFingerPosition), (0, 0, 255), 5)


        else:
            if self.isRightFingerUp == True:
                # cv2.line(self.outputFrame, (self.rightFingerPosition), (self.rightFingerPosition), (0, 255, 255), 50)
                if self.fakePlayMode:
                    if self.songMode and self.strokeArr[self.strokeIdx]:
                        sound = self.codeSounds[self.codes[self.curCode]]
                        sound.stop()
                        sound.play()

                    if self.codeMode:
                        sound = self.codeSounds[self.codes[self.curCode]]
                        sound.stop()
                        sound.play()


                if self.strokeMode or self.songMode:
                    self.strokeIdx = self.strokeIdx + 1

            self.isRightFingerUp = False

            # cv2.line(self.outputFrame, (self.rightFingerPosition), (self.rightFingerPosition), (0, 255, 0), 5)

    def detectGuitar(self):
        src = self.inputFrame[self.yCutOff:700, 0:self.rightCutOff].copy()
        canny = cv2.Canny(src, self.cannyMin, self.cannyMax, None, 3)
        lines = cv2.HoughLines(canny, 0.8, np.pi / 180, self.houghThreshold)

        if self.strokeMode:
            self.showStroke()
            self.doStroke()

        if lines is not None:
            sortedLines = []
            avgTheta = 0
            avgRho = 0
            guitarLowerY1 = -9999
            guitarUpperY1 = 9999
            guitarLowerY2 = -9999
            guitarUpperY2 = 9999

            for i in range(len(lines)):
                r, t = lines[i][0]

                avgTheta = avgTheta + t
                avgRho = avgRho + r

                sortedLines.append([r,t])

            avgTheta = avgTheta / len(lines)
            avgRho = avgRho / len(lines)

            sortedLines.sort()

            for i in sortedLines:
                r = i[0]
                t = i[1]

                if abs(t-avgTheta) > 0.09: continue

                px1, py1 = self.getIntersection(r, t, self.leftCutOff, 0)
                px2, py2 = self.getIntersection(r, t, self.rightCutOff, 0)

                if py1 > guitarLowerY1: guitarLowerY1 = py1
                if py1 < guitarUpperY1: guitarUpperY1 = py1
                if py2 > guitarLowerY2: guitarLowerY2 = py2
                if py2 < guitarUpperY2: guitarUpperY2 = py2

                self.guitarLowerY1 = int(guitarLowerY1)
                self.guitarUpperY1 = int(guitarUpperY1)
                self.guitarLowerY2 = int(guitarLowerY2)
                self.guitarUpperY2 = int(guitarUpperY2)

            if self.printGuitarEdge:
                cv2.line(self.outputFrame, (10, self.guitarLowerY1 + self.yCutOff), (10, int(self.guitarUpperY1 + self.yCutOff)), (0, 255, 255), 1)
                cv2.line(self.outputFrame, (900 - 10, self.guitarLowerY2 + self.yCutOff), (900 - 10, int(self.guitarUpperY2 + self.yCutOff)), (0, 255, 255), 1)
                cv2.line(self.outputFrame, (10, self.guitarUpperY1 + self.yCutOff), (900 - 10, int(self.guitarUpperY2 + self.yCutOff)), (0, 255, 255), 1)
                cv2.line(self.outputFrame, (10, self.guitarLowerY1 + self.yCutOff), (900 - 10, int(self.guitarLowerY2 + self.yCutOff)), (0, 255, 255), 1)

            thre = self.guitarLowerY2 - self.guitarUpperY2
            avgPy1 = (self.guitarUpperY1 + self.guitarLowerY1) / 2
            avgPy2 = (self.guitarUpperY2 + self.guitarLowerY2) / 2

            guitarLowerY1 = self.guitarLowerY1
            guitarUpperY1 = self.guitarUpperY1
            guitarLowerY2 = self.guitarLowerY2
            guitarUpperY2 = self.guitarUpperY2

            while guitarLowerY1 <= 720:
                cv2.line(canny, (0, guitarLowerY1), (900, guitarLowerY2), (0, 0, 0), 2)
                guitarLowerY1 = guitarLowerY1 + 1
                guitarLowerY2 = guitarLowerY2 + 1

            while guitarUpperY2 >= 0:
                cv2.line(canny, (0, guitarUpperY1), (900, guitarUpperY2), (0, 0, 0), 2)
                guitarUpperY1 = guitarUpperY1 - 1
                guitarUpperY2 = guitarUpperY2 - 1

            verLines = cv2.HoughLines(canny, 2.3, np.pi / 180, int(thre / 1.5))

            sortedVerLines = []
            dots = []

            if verLines is not None:
                for i in range(len(verLines)):
                    for rho, theta in verLines[i]:
                        if theta < avgTheta - 90 * math.pi / 180 - 5 * math.pi / 180 or theta > avgTheta - 90 * math.pi / 180 + 5 * math.pi / 180:
                            continue

                        sortedVerLines.append([rho, theta])

                        a = np.cos(theta)
                        b = np.sin(theta)
                        x0 = a * rho
                        y0 = b * rho
                        x1 = int(x0 + 1000 * (-b))
                        y1 = int(y0 + 1000 * (a))
                        x2 = int(x0 - 1000 * (-b))
                        y2 = int(y0 - 1000 * (a))

                        px, py = self.getIntersection(rho, theta, avgRho, avgTheta)

                        dots.append([px, py])

                        if self.printHorizontalLines:
                            cv2.line(self.outputFrame, (x1, y1), (x2, y2), (0, 255, 255), 1)

                dots.sort(reverse=True)
                dots = np.array(dots)
                dots = dots.T
                filteredDots = []
                startIdx = 0

                if len(dots) > 0:
                    for i in range(len(dots[0])-1):
                        diff = math.sqrt((dots[0][i + 1] - dots[0][i])**2 + (dots[1][i + 1] - dots[1][i])**2)

                        if diff > 10:
                            if i-startIdx > 0:
                                x = sum(dots[0][startIdx:i + 1]) / len(dots[0][startIdx:i + 1])
                                y = sum(dots[1][startIdx:i + 1]) / len(dots[1][startIdx:i + 1])
                                filteredDots.append([x, y])
                            startIdx = i + 1

                    x = sum(dots[0][startIdx:len(dots[0])]) / len(dots[0][startIdx:len(dots[0])])
                    y = sum(dots[1][startIdx:len(dots[0])]) / len(dots[1][startIdx:len(dots[0])])
                    filteredDots.append([x, y])

                    filteredDots = np.array(filteredDots)
                    removeIdx = []

                    for i in range(len(filteredDots) - 2):
                        x1 = filteredDots[i][0]
                        y1 = filteredDots[i][1]
                        x2 = filteredDots[i + 1][0]
                        y2 = filteredDots[i + 1][1]
                        x3 = filteredDots[i + 2][0]
                        y3 = filteredDots[i + 2][1]

                        d1 = self.getDistance(x1, y1, x2, y2)
                        d2 = self.getDistance(x2, y2, x3, y3)

                        if d1 > d2 * 2:
                            removeIdx.append(i)
                        else:
                            break

                    filteredDots = np.delete(filteredDots, removeIdx, 0)
                    filteredDots = filteredDots[0:len(filteredDots) - 2]
                    if len(filteredDots) > 17:
                        filteredDots = filteredDots[0:17]

                    if self.printFilteredDots:
                        for i in range(len(filteredDots)):
                            cv2.line(self.outputFrame, (int(filteredDots[i][0]), int(filteredDots[i][1] + self.yCutOff)),
                                     (int(filteredDots[i][0]), int(filteredDots[i][1] + self.yCutOff)), (255, 255, 255), 10)

                    # Fill missed dots
                    dotArray = filteredDots
                    dotArrayIdx = 1

                    while len(dotArray) < 21 and len(dotArray) > 5:

                        if dotArrayIdx == len(dotArray) - 1:
                            x1 = dotArray[dotArrayIdx - 1][0]
                            y1 = dotArray[dotArrayIdx - 1][1]
                            x2 = dotArray[dotArrayIdx][0]
                            y2 = dotArray[dotArrayIdx][1]

                            d1 = self.getDistance(x1, y1, x2, y2)

                            slope = abs((y2 - y1) / (x2 - x1))
                            angle = math.atan(slope)

                            nx = x2 - d1 * self.stringOffset[dotArrayIdx - 2] * math.cos(angle)
                            ny = y2 - d1 * self.stringOffset[dotArrayIdx - 2] * math.sin(angle)
                            dotArray = np.insert(dotArray, dotArrayIdx + 1, [nx, ny], axis=0)

                        else:
                            x1 = dotArray[dotArrayIdx - 1][0]
                            y1 = dotArray[dotArrayIdx - 1][1]
                            x2 = dotArray[dotArrayIdx][0]
                            y2 = dotArray[dotArrayIdx][1]
                            x3 = dotArray[dotArrayIdx + 1][0]
                            y3 = dotArray[dotArrayIdx + 1][1]

                            d1 = self.getDistance(x1, y1, x2, y2)
                            d2 = self.getDistance(x2, y2, x3, y3)

                            slope = abs((y2 - y1) / (x2 - x1))
                            angle = math.atan(slope)

                            if d2 < d1 * self.stringOffset[dotArrayIdx - 1] * 1.3:
                                dotArrayIdx = dotArrayIdx + 1
                            else:
                                nx = x2 - d1 * self.stringOffset[dotArrayIdx - 1] * math.cos(angle)
                                ny = y2 - d1 * self.stringOffset[dotArrayIdx - 1] * math.sin(angle)
                                dotArray = np.insert(dotArray, dotArrayIdx + 1, [nx, ny], axis=0)

                    self.stringPosition = []
                    filterNo = 0.05

                    if len(dotArray) == 21:
                        if self.printFilteredDots:
                            for i in range(len(dotArray)):
                                cv2.line(self.outputFrame, (int(dotArray[i][0]), int(dotArray[i][1]+ self.yCutOff)),
                                         (int(dotArray[i][0]), int(dotArray[i][1] + self.yCutOff)), (0, 0, 255), 5)

                        self.firstDotX = dotArray[0][0]

                        speed = 0
                        for i in range(16, 21):
                            x1 = int(dotArray[i][0])
                            y1 = int(dotArray[i][1])

                            slope = (self.guitarUpperY1 + self.guitarLowerY1) / 2 - (self.guitarUpperY2 + self.guitarLowerY2) / 2 + 0.00001
                            slope = 900 / slope

                            x2 = int(dotArray[i][0] - 1000)
                            y2 = int(dotArray[i][1] - 1000 * slope)

                            px1, py1 = self.getIntersectionDot(10, self.guitarUpperY1, 900, self.guitarUpperY2, x1, y1, x2, y2)
                            px2, py2 = self.getIntersectionDot(10, self.guitarLowerY1, 900, self.guitarLowerY2, x1, y1, x2, y2)

                            distance = self.getDistance(px1, py1, px2, py2)
                            angle = math.atan(slope)
                            stringPositionLine = []

                            d = distance * 4 / 44
                            px = px2 + d * math.cos(angle)
                            py = py2 + d * math.sin(angle)

                            if self.filterStart:
                                px = self.lastStringPosition[i - 16][0][0] * (1 - filterNo) + px * filterNo
                                py = self.lastStringPosition[i - 16][0][1] * (1 - filterNo) + py * filterNo
                            if self.printStringPositions:
                                cv2.line(self.outputFrame, (int(px), int(py + self.yCutOff)), (int(px), int(py + self.yCutOff)), (0, 255, 255), 5)
                            stringPositionLine.append([px, py])

                            d = distance * 11 / 44
                            px = px2 + d * math.cos(angle)
                            py = py2 + d * math.sin(angle)

                            if self.filterStart:
                                px = self.lastStringPosition[i - 16][1][0] * (1 - filterNo) + px * filterNo
                                py = self.lastStringPosition[i - 16][1][1] * (1 - filterNo) + py * filterNo
                                speed = speed + abs(self.lastStringPosition[i - 16][1][0] - px) + abs(
                                    self.lastStringPosition[i - 16][1][1] - py)
                            if self.printStringPositions:
                                cv2.line(self.outputFrame, (int(px), int(py + self.yCutOff)),
                                         (int(px), int(py + self.yCutOff)), (0, 255, 255), 5)
                            stringPositionLine.append([px, py])

                            d = distance * 18 / 44
                            px = px2 + d * math.cos(angle)
                            py = py2 + d * math.sin(angle)

                            if self.filterStart:
                                px = self.lastStringPosition[i - 16][2][0] * (1 - filterNo) + px * filterNo
                                py = self.lastStringPosition[i - 16][2][1] * (1 - filterNo) + py * filterNo
                                speed = speed + abs(self.lastStringPosition[i - 16][1][0] - px) + abs(
                                    self.lastStringPosition[i - 16][1][1] - py)
                            if self.printStringPositions:
                                cv2.line(self.outputFrame, (int(px), int(py + self.yCutOff)),
                                         (int(px), int(py + self.yCutOff)), (0, 255, 255), 5)
                            stringPositionLine.append([px, py])

                            d = distance * 25 / 44
                            px = px2 + d * math.cos(angle)
                            py = py2 + d * math.sin(angle)

                            if self.filterStart:
                                px = self.lastStringPosition[i - 16][3][0] * (1 - filterNo) + px * filterNo
                                py = self.lastStringPosition[i - 16][3][1] * (1 - filterNo) + py * filterNo
                                speed = speed + abs(self.lastStringPosition[i - 16][1][0] - px) + abs(
                                    self.lastStringPosition[i - 16][1][1] - py)
                            if self.printStringPositions:
                                cv2.line(self.outputFrame, (int(px), int(py + self.yCutOff)),
                                         (int(px), int(py + self.yCutOff)), (0, 255, 255), 5)
                            stringPositionLine.append([px, py])

                            d = distance * 32 / 44
                            px = px2 + d * math.cos(angle)
                            py = py2 + d * math.sin(angle)

                            if self.filterStart:
                                px = self.lastStringPosition[i - 16][4][0] * (1 - filterNo) + px * filterNo
                                py = self.lastStringPosition[i - 16][4][1] * (1 - filterNo) + py * filterNo
                                speed = speed + abs(self.lastStringPosition[i - 16][1][0] - px) + abs(
                                    self.lastStringPosition[i - 16][1][1] - py)
                            if self.printStringPositions:
                                cv2.line(self.outputFrame, (int(px), int(py + self.yCutOff)),
                                         (int(px), int(py + self.yCutOff)), (0, 255, 255), 5)
                            stringPositionLine.append([px, py])

                            d = distance * 39 / 44
                            px = px2 + d * math.cos(angle)
                            py = py2 + d * math.sin(angle)

                            if self.filterStart:
                                px = self.lastStringPosition[i - 16][5][0] * (1 - filterNo) + px * filterNo
                                py = self.lastStringPosition[i - 16][5][1] * (1 - filterNo) + py * filterNo
                                speed = speed + abs(self.lastStringPosition[i - 16][1][0] - px) + abs(
                                    self.lastStringPosition[i - 16][1][1] - py)
                            if self.printStringPositions:
                                cv2.line(self.outputFrame, (int(px), int(py + self.yCutOff)),
                                         (int(px), int(py + self.yCutOff)), (0, 255, 255), 5)
                            stringPositionLine.append([px, py])

                            self.stringPosition.append(stringPositionLine)

                        self.lastStringPosition = self.stringPosition

                        if self.filterStart is False:
                            self.filterStart = True


                        if self.codeMode:
                            if self.isLeft:
                                self.codeJudge()

                            if speed < 800:
                                self.drawCode()

                        if self.songMode:
                            self.doSong()