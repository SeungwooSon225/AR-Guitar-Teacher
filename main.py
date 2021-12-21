import cv2
import threading
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from queue import Queue
from FrameMaker import FrameMaker
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
from PyQt5.uic import loadUi
import playsound
import pygame

running = False
q = Queue()

fm = FrameMaker()

# def run(q):
#     global running
#     cap = cv2.VideoCapture('test.mp4')
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     label.resize(width, height)
#     while running:
#         ret, img = cap.read()
#
#         if q.qsize() > 0:
#             code = q.get()
#             fm.setCode(code)
#
#         fm.setFrame(img)
#         fm.canny()
#         fm.detectHand()
#         fm.detectGuitar()
#         frame = fm.getFrame()
#
#         if ret:
#             img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             h,w,c = img.shape
#             qImg = QtGui.QImage(img.data, w, h, w*c, QtGui.QImage.Format_RGB888)
#             pixmap = QtGui.QPixmap.fromImage(qImg)
#             label.setPixmap(pixmap)
#         else:
#             QtWidgets.QMessageBox.about(win, "Error", "Cannot read frame.")
#             print("cannot read frame.")
#             break
#     cap.release()
#     print("Thread end.")
#
# # q.put("D")
# # if q.qsize() > 0:
# #     print(q.get())
#
# def stop():
#     global running
#     q.put("D")
#
#
# def start():
#     global running
#     running = True
#     th = threading.Thread(target=run, args=(q,))
#     th.start()
#     print("started..")
#
# def onExit():
#     print("exit")
#     stop()
#
# app = QtWidgets.QApplication([])
# # win = QtWidgets.QWidget()
# # vbox = QtWidgets.QVBoxLayout()
# # label = QtWidgets.QLabel()
# # btn_start = QtWidgets.QPushButton("Camera On")
# # btn_stop = QtWidgets.QPushButton("Camera Off")
# # vbox.addWidget(label)
# # vbox.addWidget(btn_start)
# # vbox.addWidget(btn_stop)
# # win.setLayout(vbox)
#

class MyApp(QDialog):
    q = Queue()
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = loadUi("./ui/myapp.ui", self)
        self.pushButton_G.setStyleSheet(
            '''
            QPushButton{image:url(./icon/Gcode.png); border:0px;}
            QPushButton:hover{image:url(./icon/Gcode_gray.png); border:0px;}
            ''')

        self.pushButton_D.setStyleSheet(
        '''
        QPushButton{image:url(./icon/Dcode.png); border:0px;}
        QPushButton:hover{image:url(./icon/Dcode_gray.png); border:0px;}
        ''')

        self.pushButton_Em.setStyleSheet(
            '''
            QPushButton{image:url(./icon/Emcode.png); border:0px;}
            QPushButton:hover{image:url(./icon/Emcode_gray.png); border:0px;}
            ''')

        self.pushButton_Bm.setStyleSheet(
            '''
            QPushButton{image:url(./icon/Bmcode.png); border:0px;}
            QPushButton:hover{image:url(./icon/Bmcode_gray.png); border:0px;}
            ''')

        self.pushButton_C.setStyleSheet(
            '''
            QPushButton{image:url(./icon/Ccode.png); border:0px;}
            QPushButton:hover{image:url(./icon/Ccode_gray.png); border:0px;}
            ''')

        self.pushButton_G7.setStyleSheet(
            '''
            QPushButton{image:url(./icon/G7code.png); border:0px;}
            QPushButton:hover{image:url(./icon/G7code_gray.png); border:0px;}
            ''')


        self.pushButton_Am.setStyleSheet(
            '''
            QPushButton{image:url(./icon/Amcode.png); border:0px;}
            QPushButton:hover{image:url(./icon/Amcode_gray.png); border:0px;}
            ''')

        self.pushButton_back.setStyleSheet(
            '''
            QPushButton{image:url(./icon/back.png); border:0px;}
            QPushButton:hover{image:url(./icon/back_gray.png); border:0px;}
            ''')

        self.pushButton_train.setStyleSheet(
            '''
            QPushButton{image:url(./icon/trainingOn.png); border:0px;}
            QPushButton:hover{image:url(./icon/trainingOn_gray.png); border:0px;}
            ''')

        self.pushButton_play.setStyleSheet(
            '''
            QPushButton{image:url(./icon/play.png); border:0px;}
            QPushButton:hover{image:url(./icon/play_gray.png); border:0px;}
            ''')

        self.codeImage.setStyleSheet('image:url(./icon/Gcode.png); border:0px;')
        self.backImage.setStyleSheet('image:url(./icon/backboard.png); border:0px;')
        self.boardImage.setStyleSheet('image:url(./icon/board.png); border:0px;')

        pygame.mixer.init()
        self.gSound = pygame.mixer.Sound('./sound/g.mp3')
        self.dSound = pygame.mixer.Sound('./sound/d.mp3')
        self.emSound = pygame.mixer.Sound('./sound/em.mp3')
        self.bmSound = pygame.mixer.Sound('./sound/bm.mp3')
        self.cSound = pygame.mixer.Sound('./sound/c.mp3')
        self.g7Sound = pygame.mixer.Sound('./sound/g7.mp3')
        self.amSound = pygame.mixer.Sound('./sound/am.mp3')

        self.sounds = [self.gSound, self.dSound, self.emSound, self.bmSound, self.cSound, self.g7Sound, self.amSound]
        self.codes = {'G': 0, 'D': 1, 'Em': 2, 'Bm': 3, 'C': 4, 'G7': 5, 'Am': 6}
        self.code = 'G'

    def threadStart(self):
        global running
        running = True
        th = threading.Thread(target=self.run, args=(q,))
        th.start()
        print("started..")

    def train(self):
        fm.setFakePlayMode()

        if fm.getFakePlayMode():
            self.pushButton_train.setStyleSheet(
                '''
                QPushButton{image:url(./icon/trainingOff.png); border:0px;}
                QPushButton:hover{image:url(./icon/trainingOff_gray.png); border:0px;}
                ''')
        else:
            self.pushButton_train.setStyleSheet(
                '''
                QPushButton{image:url(./icon/trainingOn.png); border:0px;}
                QPushButton:hover{image:url(./icon/trainingOn_gray.png); border:0px;}
                ''')



    def stop(self):
        global running
        running = False
        widget.setCurrentIndex(0)

    def btnDownG(self):
        q.put("G")
        self.code = 'G'
        self.codeImage.setStyleSheet('image:url(./icon/Gcode.png); border:0px;')

    def btnDownD(self):
        q.put("D")
        self.code = 'D'
        self.codeImage.setStyleSheet('image:url(./icon/Dcode.png); border:0px;')

    def btnDownEm(self):
        q.put("Em")
        self.code = 'Em'
        self.codeImage.setStyleSheet('image:url(./icon/Emcode.png); border:0px;')

    def btnDownBm(self):
        q.put("Bm")
        self.code = 'Bm'
        self.codeImage.setStyleSheet('image:url(./icon/Bmcode.png); border:0px;')

    def btnDownC(self):
        q.put("C")
        self.code = 'C'
        self.codeImage.setStyleSheet('image:url(./icon/Ccode.png); border:0px;')

    def btnDownG7(self):
        q.put("G7")
        self.code = 'G7'
        self.codeImage.setStyleSheet('image:url(./icon/G7code.png); border:0px;')

    def btnDownAm(self):
        q.put("Am")
        self.code = 'Am'
        self.codeImage.setStyleSheet('image:url(./icon/Amcode.png); border:0px;')

    def btnDownPlaySound(self):
        sound = self.sounds[self.codes[self.code]]
        sound.stop()
        sound.play()


    def run(self, q):
        global running
        # cap = cv2.VideoCapture('test.mp4')
        cap = cv2.VideoCapture(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.ui.label.resize(width, height)

        while running:
            ret, img = cap.read()

            if q.qsize() > 0:
                code = q.get()
                fm.setCode(code)

            fm.setCodeMode()
            fm.setFrame(img)
            fm.canny()
            fm.detectHand()
            fm.detectGuitar()
            frame = fm.getFrame()

            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.ui.label.setPixmap(pixmap)
            else:
                print("cannot read frame.")
                break
        cap.release()
        print("Thread end.")

class MainWindow(QDialog):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = loadUi("./ui/main.ui", self)
        # self.pushButton.setStyleSheet('image:url(./icon/button.png); border:0px;')
        self.pushButton.setStyleSheet(
            '''
            QPushButton{image:url(./icon/button.png); border:0px;}
            QPushButton:hover{image:url(./icon/button_gray.png); border:0px;}
            ''')

        self.pushButton_stroke.setStyleSheet(
            '''
            QPushButton{image:url(./icon/buttonStroke.png); border:0px;}
            QPushButton:hover{image:url(./icon/buttonStroke_gray.png); border:0px;}
            ''')

        self.pushButton_song.setStyleSheet(
            '''
            QPushButton{image:url(./icon/buttonSong.png); border:0px;}
            QPushButton:hover{image:url(./icon/buttonSong_gray.png); border:0px;}
            ''')

    def go(self):
        me.threadStart()
        widget.setCurrentIndex(1)

    def openStrokeWindow(self):
        strokeWindow.threadStart()
        widget.setCurrentIndex(2)

    def openSongWindow(self):
        songWindow.threadStart()
        widget.setCurrentIndex(3)

class StrokeWindow(QDialog):
    q = Queue()
    def __init__(self):
        super(StrokeWindow, self).__init__()
        self.ui = loadUi("./ui/stroke.ui", self)
        self.strokeImage.setStyleSheet('image:url(./icon/stroke.png); border:0px;')
        self.pushButton_back.setStyleSheet(
            '''
            QPushButton{image:url(./icon/back.png); border:0px;}
            QPushButton:hover{image:url(./icon/back_gray.png); border:0px;}
            ''')

        self.pushButton_do.setStyleSheet(
            '''
            QPushButton{image:url(./icon/strokeTrain.png); border:0px;}
            QPushButton:hover{image:url(./icon/strokeTrain_gray.png); border:0px;}
            ''')

        self.pushButton_show.setStyleSheet(
            '''
            QPushButton{image:url(./icon/strokePlay.png); border:0px;}
            QPushButton:hover{image:url(./icon/strokePlay_gray.png); border:0px;}
            ''')

        self.background.setStyleSheet('image:url(./icon/board2.png); border:0px;')

    def goBack(self):
        global running
        running = False
        widget.setCurrentIndex(0)

    def showStroke(self):
        fm.setShowStroke()

        if fm.getShowStroke():
            self.pushButton_show.setStyleSheet(
                '''
                QPushButton{image:url(./icon/strokeOff.png); border:0px;}
                QPushButton:hover{image:url(./icon/strokeOff_gray.png); border:0px;}
                ''')
        else:
            self.pushButton_show.setStyleSheet(
                '''
                QPushButton{image:url(./icon/strokePlay.png); border:0px;}
                QPushButton:hover{image:url(./icon/strokePlay_gray.png); border:0px;}
                ''')

    def doStroke(self):
        fm.setDoStroke()

        if fm.getDoStroke():
            self.pushButton_do.setStyleSheet(
                '''
                QPushButton{image:url(./icon/strokeTrainOff.png); border:0px;}
                QPushButton:hover{image:url(./icon/strokeTrainOff_gray.png); border:0px;}
                ''')
        else:
            self.pushButton_do.setStyleSheet(
                '''
                QPushButton{image:url(./icon/strokeTrain.png); border:0px;}
                QPushButton:hover{image:url(./icon/strokeTrain_gray.png); border:0px;}
                ''')

    def threadStart(self):
        global running
        running = True
        th = threading.Thread(target=self.run, args=(q,))
        th.start()
        print("started..")

    def run(self, q):
        global running
        # cap = cv2.VideoCapture('test.mp4')
        cap = cv2.VideoCapture(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.ui.label.resize(width, height)

        while running:
            ret, img = cap.read()

            fm.setStrokeMode()
            fm.setFrame(img)
            fm.canny()
            fm.detectHand()
            fm.detectGuitar()
            frame = fm.getFrame()

            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.ui.label.setPixmap(pixmap)
            else:
                print("cannot read frame.")
                break
        cap.release()
        print("Thread end.")

class SongWindow(QDialog):
    q = Queue()
    def __init__(self):
        super(SongWindow, self).__init__()
        self.ui = loadUi("./ui/song.ui", self)
        self.pushButton_back.setStyleSheet(
            '''
            QPushButton{image:url(./icon/back.png); border:0px;}
            QPushButton:hover{image:url(./icon/back_gray.png); border:0px;}
            ''')
        self.pushButton_do.setStyleSheet(
            '''
            QPushButton{image:url(./icon/songTrain.png); border:0px;}
            QPushButton:hover{image:url(./icon/songTrain_gray.png); border:0px;}
            ''')

        self.pushButton_train.setStyleSheet(
            '''
                QPushButton{image:url(./icon/trainingOn.png); border:0px;}
                QPushButton:hover{image:url(./icon/trainingOn_gray.png); border:0px;}
                ''')

        self.backImage.setStyleSheet('image:url(./icon/backboard.png); border:0px;')
        self.background.setStyleSheet('image:url(./icon/board3.png); border:0px;')
        self.score.setStyleSheet('image:url(./icon/score1.png); border:0px;')

    def goBack(self):
        global running
        running = False
        widget.setCurrentIndex(0)

    def threadStart(self):
        global running
        running = True
        th = threading.Thread(target=self.run, args=(q,))
        th.start()
        print("started..")

    def doSong(self):
        fm.setDoSong()

    def train(self):
        fm.setFakePlayMode()

        print(fm.getFakePlayMode())
        if fm.getFakePlayMode():
            self.pushButton_train.setStyleSheet(
                '''
                QPushButton{image:url(./icon/trainingOff.png); border:0px;}
                QPushButton:hover{image:url(./icon/trainingOff_gray.png); border:0px;}
                ''')
        else:
            self.pushButton_train.setStyleSheet(
                '''
                QPushButton{image:url(./icon/trainingOn.png); border:0px;}
                QPushButton:hover{image:url(./icon/trainingOn_gray.png); border:0px;}
                ''')

    def run(self, q):
        global running
        # cap = cv2.VideoCapture('test.mp4')
        cap = cv2.VideoCapture(0)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.ui.label.resize(width, height)

        code = ''

        while running:
            ret, img = cap.read()

            curCode, idx = fm.getCode()
            if code != curCode:
                code = curCode
                self.codeImage.setStyleSheet('image:url(./icon/'+code+'code.png); border:0px;')
                self.score.setStyleSheet('image:url(./icon/score' + str(idx+1) + '.png); border:0px;')

            fm.setSongMode()
            fm.setFrame(img)
            fm.canny()
            fm.detectHand()
            fm.detectGuitar()
            frame = fm.getFrame()

            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, c = img.shape
                qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qImg)
                self.ui.label.setPixmap(pixmap)
            else:
                print("cannot read frame.")
                break
        cap.release()
        print("Thread end.")

app = QtWidgets.QApplication(sys.argv)

widget = QtWidgets.QStackedWidget()

me = MyApp()
main = MainWindow()
strokeWindow = StrokeWindow()
songWindow = SongWindow()

widget.addWidget(main)
widget.addWidget(me)
widget.addWidget(strokeWindow)
widget.addWidget(songWindow)

widget.setFixedHeight(970)
widget.setFixedWidth(1500)
widget.show()

sys.exit(app.exec())