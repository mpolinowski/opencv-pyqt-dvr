import cv2
import os
import sys

from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout
from PyQt6.uic import loadUiType


# prepare live stream
RTSP_URL = 'rtsp://admin:instar@192.168.2.120/livestream/13'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

# load ui definition from Qt designer
ui, _ = loadUiType('opencv-pyqt-dvr.ui')

# initialize application
class DvrDashboard(QMainWindow, ui):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle("INSTAR Motion Detector")
        self.setupUi(self)
        # handle clicks on button "motion_detection"
        self.motion_detection.clicked.connect(self.get_livestream)
        # handle clicks on button "exit"
        self.exit.clicked.connect(self.close_window)


    def get_livestream(self):
        print("INFO :: Connecting to IP Camera")
        # take still frame to feed the live video
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print('ERROR :: Cannot open RTSP stream')
            exit(-1)

        while True:
            # get frame from livestream
            success, img = cap.read()
            # get second frame for motion detection
            _, val_img = cap.read()
            # get absolute difference between both frames
            delta = cv2.absdiff(img, val_img)
            # find contours in delta for moving object location
            grayscale  = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(grayscale, (5,5), 0)
            canny = cv2.Canny(blur, 35, 75)
            dilated = cv2.dilate(canny, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for ctr in contours:
                # filter small objects
                if cv2.contourArea(ctr) < 5000:
                    continue
                # get bounding box location
                x,y,w,h = cv2.boundingRect(ctr)
                # draw bounding box around object
                cv2.rectangle(img, (x,y), (x+w, y+h), (204,119,0), 2)
                # get canny image
                cv2.imwrite('detection_object_contour.jpg', canny)
                # get marked image
                cv2.imwrite('detection_bounding_box.jpg', img)

                # display detection in frontend
                detection_contour = QImage('detection_object_contour.jpg')
                detection_contour_map = QPixmap.fromImage(detection_contour)
                self.detection_contour.setPixmap(detection_contour_map)

                detection_image = QImage('detection_bounding_box.jpg')
                detection_image_map = QPixmap.fromImage(detection_image)
                self.detection.setPixmap(detection_image_map)
                
            # also show live video for reference
            cv2.imshow(RTSP_URL, img)

            if cv2.waitKey(1) == 27:  # Keep running until you press `esc`
                break

        cap.release()
        cv2.destroyAllWindows()


    def close_window(self):
        print("WARNING :: Application shutdown")
        self.close()

# execute the Qt window
def main():
    app = QApplication(sys.argv)
    window = DvrDashboard()
    window.show()
    app.exec()

# start
if __name__ == '__main__':
    main()