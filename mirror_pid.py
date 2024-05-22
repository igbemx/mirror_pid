"""
File: mirror_pid.py

Camera image based mirror PID regulator.

Author: Igor Beinik
Date: 2024-05-22
"""

import sys
import time
import cv2
import threading
import queue
import numpy as np
import tango
# from simple_pid import PID
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QFormLayout, QMenu, QLabel, QLineEdit, QAction, QMessageBox, QPushButton, QWidget
from pyqtgraph import ImageView, LineSegmentROI, ROI, RectROI, TextItem, ViewBox, PlotWidget, PlotDataItem, ScatterPlotItem
from PyQt5.QtCore import QTimer, QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QDoubleValidator, QIntValidator
from scipy.optimize import curve_fit

from skimage.filters import gaussian
from skimage.measure import regionprops
from sklearn.linear_model import LinearRegression

app = QApplication(sys.argv)


class CustomImageView(ImageView):
    def __init__(self):
        super(CustomImageView, self).__init__()
        self.initContextMenu()
        self.update_levels = True
        self.update_range = False

    def initContextMenu(self):
        self.contextMenu = QMenu(self)
        autoLevels = QAction("Auto levels", self)
        autoLevels.triggered.connect(self.customMenuItemClicked)
        autoLevels.setCheckable(True)
        autoLevels.setChecked(True)
        autoLevels.triggered.connect(self.auto_levels_action)
        self.contextMenu.addAction(autoLevels)

    def auto_levels_action(self, checked):
        if checked:
            self.update_levels = True
        else:
            self.update_levels = False

    def customMenuItemClicked(self):
        pass
        # QMessageBox.information(self, "Auto levels Clicked", "Auto levels clicked!")

    def contextMenuEvent(self, event):
        # Show the context menu at the cursor's position
        self.contextMenu.exec_(self.mapToGlobal(event.pos()))


class LEDButton(QWidget):
    clicked = pyqtSignal()

    def __init__(self, text1, text2, led_on=False):
        super().__init__()
        self.text1 = text1
        self.text2 = text2
        self.button = QPushButton(text1)
        self.led_label = QLabel()
        self.led_on = led_on

        self.initUI()

    def initUI(self):
        layout = QHBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.led_label)

        self.setLayout(layout)
        self.update_led()
        self.button.clicked.connect(self.on_button_clicked)

    def on_button_clicked(self):
        self.clicked.emit()
        self.toggle_led()
        current_text = self.button.text()
        self.button.setText(self.text2 if current_text == self.text1 else self.text1)

    def toggle_led(self):
        self.led_on = not self.led_on
        self.update_led()

    def update_led(self):
        color = "green" if self.led_on else "red"
        self.led_label.setStyleSheet(f"background-color: {color}; border-radius: 10px; min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;")


class PIDController:
    def __init__(self, P=1.0, I=0.5, D=0.001, setpoint=0):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.setpoint = setpoint
        self.sample_time = 0.01  # 10 ms

        self._clear()

    def _clear(self):
        """Clears PID computations and coefficients"""
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.windup_guard = 20.0
        self.output = 0.0

    def update(self, feedback_value):
        """Calculates PID value for given reference feedback
        """
        error = self.setpoint - feedback_value
        delta_error = error - self.last_error

        self.PTerm = self.Kp * error
        self.ITerm += error * self.sample_time

        # # Windup Guard
        # if self.ITerm < -self.windup_guard:
        #     self.ITerm = -self.windup_guard
        # elif self.ITerm > self.windup_guard:
        #     self.ITerm = self.windup_guard

        self.DTerm = 0.0
        if self.sample_time > 0:
            self.DTerm = delta_error / self.sample_time

        # Remember last error for next time
        self.last_error = error

        self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Set integral windup guard"""
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """Set the period, in seconds, at which the calculation is performed"""
        self.sample_time = sample_time

    def setSetpoint(self, setpoint):
        """Initializes the setpoint of PID"""
        self.setpoint = setpoint

    def getOutput(self):
        """Returns the current output of the PID controller"""
        return self.output


class MainWidget(QMainWindow):
    def __init__(self, stream_url=None, mirror_motor=None):
        super(MainWidget, self).__init__()

        self.stream_url = stream_url
        self.frame_queue = queue.Queue(maxsize=1)
        self.mirror_alias = mirror_motor
        self.gray_frame = None
        self.source1_mean = None
        self.source2_mean = None
        self.mirror_setp = None
        self.update_rate = 300
        self.mirror_envelop = 50
        self.regulator_on = False
        self.Kp = 0.18
        self.Ki = 0.001
        self.Kd = 0.0
        self.feedback = 0
        self.feedback_K = 6.579445
        self.feedback_offset = 1666.25329

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.plot_layout = QVBoxLayout()
        self.view = CustomImageView()
        self.view.roi.setSize(QPointF(50.0, 100.0))
        self.view.roi.setPos(860, 400)
        self.src_2_label = TextItem("Source 2", anchor=(0.5, 0.5), color='y')
        self.src_2_label.setParentItem(self.view.roi)
        self.src_2_label.setPos(30, -10)

        self.view.view.setMouseMode(ViewBox.RectMode)
        self.view.roi.sigRegionChanged.connect(self.on_view_roi_selected)
        self.plot_layout.addWidget(self.view)

        # Add a Line ROI
        self.line_roi = LineSegmentROI([[850, 500], [920, 500]], pen='r')
        self.view.addItem(self.line_roi)
        self.line_roi.sigRegionChanged.connect(self.update_gaussian_fit)
        self.src_1_label = TextItem("Source 1", anchor=(0, 0), color='y')
        self.src_1_label.setParentItem(self.line_roi)
        self.src_1_label.setPos(820, 500)

        # Adding control elements
        self.control_layout = QFormLayout()
        self.toggle_acquisition_button = LEDButton("Start Acquisition", "Stop Acquisition")
        self.toggle_acquisition_button.clicked.connect(self.toggle_acquisition)
        self.control_layout.addWidget(self.toggle_acquisition_button)

        self.startRegButton = LEDButton("Start Regulation", "Stop Regulation")
        self.control_layout.addWidget(self.startRegButton)
        self.startRegButton.clicked.connect(self.toggle_regulation)

        self.source1Label = QLabel('Source 1')
        self.control_layout.addRow("Source 1:", self.source1Label)

        self.source2Label = QLabel('Source 2')
        self.control_layout.addRow("Source 2:", self.source2Label)

        self.mirrorLabel = QLabel('Mirror Position')
        self.control_layout.addRow(f"Mirror Position, \u00B5rad:", self.mirrorLabel)

        self.feedbackLabel = QLabel('Feedback')
        self.control_layout.addRow("Feedback: ", self.feedbackLabel)

        self.setpEdit = QLineEdit()
        self.setpEdit.setValidator(QDoubleValidator())
        self.setpEdit.returnPressed.connect(self.update_setp)
        self.control_layout.addRow("Mirror Setpoint, \u00B5rad:", self.setpEdit)

        self.get_setp_button = QPushButton("Fetch Setpoint")
        self.control_layout.addWidget(self.get_setp_button)
        self.get_setp_button.clicked.connect(self.catch_setp)

        self.updateRateEdit = QLineEdit()
        self.updateRateEdit.setValidator(QIntValidator())
        self.updateRateEdit.returnPressed.connect(self.change_update_rate)
        self.updateRateEdit.setText(f"{self.update_rate}")
        self.control_layout.addRow("Update rate, ms:", self.updateRateEdit)

        self.mirrEnvelopEdit = QLineEdit()
        self.mirrEnvelopEdit.setValidator(QDoubleValidator())
        self.mirrEnvelopEdit.returnPressed.connect(self.update_mirror_envelop)
        self.mirrEnvelopEdit.setText(f"{self.mirror_envelop}")
        self.control_layout.addRow("Mirror Envelop, \u00B5rad:", self.mirrEnvelopEdit)

        self.pidKpEdit = QLineEdit()
        self.pidKpEdit.setValidator(QDoubleValidator())
        self.pidKpEdit.returnPressed.connect(self.pidKp_update)
        self.pidKpEdit.setText(f"{self.Kp}")
        self.control_layout.addRow("Kp: ", self.pidKpEdit)

        self.pidKiEdit = QLineEdit()
        self.pidKiEdit.setValidator(QDoubleValidator())
        self.pidKiEdit.returnPressed.connect(self.pidKi_update)
        self.pidKiEdit.setText(f"{self.Ki}")
        self.control_layout.addRow("Ki: ", self.pidKiEdit)

        self.pidKdEdit = QLineEdit()
        self.pidKdEdit.setValidator(QDoubleValidator())
        self.pidKdEdit.returnPressed.connect(self.pidKd_update)
        self.pidKdEdit.setText(f"{self.Kd}")
        self.control_layout.addRow("Kd: ", self.pidKdEdit)

        self.feedbackKEdit = QLineEdit()
        self.feedbackKEdit.setValidator(QDoubleValidator())
        self.feedbackKEdit.returnPressed.connect(self.feedback_K_update)
        self.feedbackKEdit.setText(f"{self.feedback_K}")
        self.control_layout.addRow("Feedback K: ", self.feedbackKEdit)

        self.feedbackOffsetEdit = QLineEdit()
        self.feedbackOffsetEdit.setValidator(QDoubleValidator())
        self.feedbackOffsetEdit.returnPressed.connect(self.feedback_offset_update)
        self.feedbackOffsetEdit.setText(f"{self.feedback_offset}")
        self.control_layout.addRow("Feedback Offset: ", self.feedbackOffsetEdit)

        self.getFeedbackOffsetBtn = QPushButton("Adjust Offset")
        self.control_layout.addWidget(self.getFeedbackOffsetBtn)
        self.getFeedbackOffsetBtn.clicked.connect(self.get_feedback_offset)

        # Create a plot for displaying the Gaussian fit
        self.fit_plot = PlotWidget(title="Source 1")
        self.plot_layout.addWidget(self.fit_plot)

        self.layout.addLayout(self.plot_layout)
        self.layout.addLayout(self.control_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.video_capture = cv2.VideoCapture(self.stream_url)
        self.video_thread = threading.Thread(target=self.camera_thread, args=(self.video_capture, self.frame_queue), daemon=True).start()
        try:
            self.mirror = tango.DeviceProxy(self.mirror_alias)
            self.mirror_init_pos = self.mirror.position
            print("Initial mirror position: ", self.mirror_init_pos)
        except tango.DevFailed as e:
            print("Error in Tango device connection: ", e)
            self.mirror = None
        
        self.pid = PIDController(P=0.180, I=0.001, D=0.0, setpoint=self.mirror_setp)
        self.pid.setSampleTime(self.update_rate*0.0005)  # 50 of the update time in seconds

    def camera_thread(self, cap, frame_queue):
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                frame_queue.get()
                frame_queue.put(frame)

    def update_gaussian_fit(self):
        # Extract the line data from the ROI on the image
        line_data = self.line_roi.getArrayRegion(self.gray_frame, self.view.imageItem)
        # print(self.line_roi.state)

        # Generate x values corresponding to the line_data length
        x = np.arange(len(line_data))
        # print('x: ', x)
        # print('line_data: ', line_data)
        # print('Index of maximum: ', np.argmax(line_data))

        # Fit Gaussian to the line data
        try:
            params, _ = curve_fit(self.gaussian, x, line_data, p0=[line_data.max(), len(line_data) / 2, len(line_data) / 4])
            _, self.source1_mean, _ = params
            self.source1Label.setText(f"{self.source1_mean:.3f}")
            self.fit_plot.clear()
            self.fit_plot.plot(x, line_data, pen='b')
            self.fit_plot.plot(x, self.gaussian(x, *params), pen='r')
        except Exception as e:
            print("Error in Gaussian fitting:", e)

    def gaussian(self, x, amplitude, mean, stddev):
        # print('Gaussian mean pos: ', mean)
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

    def on_view_roi_selected(self, plot_widget):
        """Plots the Gaussian fit to the ROI plot."""

        # Get the ROI mask.
        rect_data = plot_widget.getArrayRegion(self.gray_frame, self.view.imageItem)
        rect_data_prj = np.mean(rect_data, axis=1)
        x = np.arange(len(rect_data_prj))
        params, _ = curve_fit(self.gaussian, x, rect_data_prj, p0=[rect_data_prj.max(), len(rect_data_prj) / 2, len(rect_data_prj) / 4])
        _, self.source2_mean, _ = params
        self.source2Label.setText(f"{self.source2_mean:.3f}")

        self.view.ui.roiPlot.clear()
        self.view.ui.roiPlot.plot(x, rect_data_prj, pen='b')
        self.view.ui.roiPlot.plot(x, self.gaussian(x, *params), pen='r')

    def toggle_acquisition(self):
        if not self.toggle_acquisition_button.led_on:
            self.timer.start(self.update_rate)
        else:
            self.timer.stop()

    def catch_setp(self):
        self.mirror_setp = self.feedback
        self.setpEdit.setText(f"{self.mirror_setp:.3f}")
        self.pid.setSetpoint(self.mirror_setp)
        print(f"The mirror setpoint is {self.mirror_setp} now.")

    def toggle_auto_scaling(self):
        self.view.autoRangeEnabled = not self.view.autoRangeEnabled

    def toggle_regulation(self):
        if not self.startRegButton.led_on:
            self.regulator_on = True
            print("Regulator turned ON")
        else:
            self.regulator_on = False
            print("Regulator turned OFF")

    def change_update_rate(self):
        self.update_rate = int(self.updateRateEdit.text())
        print(f"New update rate is: {self.update_rate} ms.")
        self.timer.stop()
        self.timer.start(self.update_rate)
        self.pid.setSampleTime(self.update_rate*0.0005) # 50 of the update time
        print(f"Updated PID cycle time is: {self.pid.sample_time} s.")

    def update_setp(self):
        self.mirror_setp = float(self.setpEdit.text())
        self.pid.setSetpoint(self.mirror_setp)
        print("Updated mirror setpoint is: ", self.mirror_setp)

    def update_mirror_envelop(self):
        self.mirror_envelop = float(self.mirrEnvelopEdit.text())
        print("New mirror envelop is: ", self.mirror_envelop)

    def pidKp_update(self):
        self.Kp = float(self.pidKpEdit.text())
        self.pid.setKp(self.Kp)
        print("Updated Kp is: ", self.Kp)

    def pidKi_update(self):
        self.Ki = float(self.pidKiEdit.text())
        self.pid.setKi(self.Ki)
        print("Updated Ki is: ", self.Ki)

    def pidKd_update(self):
        self.Kd = float(self.pidKdEdit.text())
        self.pid.setKd(self.Kd)
        print("Updated Kd is: ", self.Kd)

    def feedback_K_update(self):
        self.feedback_K = float(self.feedbackKEdit.text())
        print("Updated Feedback K is: ", self.feedback_K)

    def feedback_offset_update(self):
        self.feedback_offset = float(self.feedbackOffsetEdit.text())
        print("Updated Feedback Offset is: ", self.feedback_offset)

    def _wait_and_process(self, duration):
        start_time = time.time()
        while True:
            time_now = time.time()
            elapsed_time = time_now - start_time
            QApplication.processEvents()
            if elapsed_time > duration:
                break

    def get_feedback_offset(self):
        first_sensor_pos = self.source1_mean
        if self.mirror:
            first_mirror_pos = self.mirror.position
        else:
            print('Could not obtain the initial mirror position..')
            first_mirror_pos = None
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle('Mirror move request!')
        msg_box.setText('The mirror needs to be moved by the envelop value in the positive direction. The procedure takes a couple of minutes. Is it OK to move it???')
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg_box.setDefaultButton(QMessageBox.Ok)

        sensor_positions = []
        mirror_positions = []
        requested_mirror_positions = np.linspace(self.mirror.position - self.mirror_envelop / 2, self.mirror.position + self.mirror_envelop / 2, 20)

        retval = msg_box.exec_()
        if retval == QMessageBox.Ok:
            self.mirror.position = requested_mirror_positions[0]
            self._wait_and_process(5)
            for pos in requested_mirror_positions:
                self.mirror.position = pos
                self._wait_and_process(3)
                while self.mirror.state() == tango.DevState.MOVING:
                    time.sleep(0.1)
                sensor_positions.append(self.source1_mean)
                mirror_positions.append(self.mirror.position)
            for p in zip(mirror_positions, sensor_positions):
                print("Acquired positions: ", p)
            QMessageBox.information(self, "Success!", "Values updated successfully! Move the mirror back to its original position!")
            self.mirror.position = first_mirror_pos
        elif retval == QMessageBox.Cancel:
            pass

        mirror_positions = np.array(mirror_positions)
        sensor_positions = np.array(sensor_positions).reshape((-1, 1))
        linear_fit = LinearRegression().fit(sensor_positions, mirror_positions)
        self.feedbackOffsetEdit.setText(str(linear_fit.intercept_))
        self.feedbackKEdit.setText(str(linear_fit.coef_[0]))
        self.feedback_offset_update()
        self.feedback_K_update()

    def update_frame(self):
        # try:
        #     ret, frame = self.video_capture.read()
        # except Exception as e:
        #     print("Update frame exception: ", e)
        #     self.video_capture = cv2.VideoCapture(self.stream_url)

        # ret, frame = self.video_capture.read()

        if not self.frame_queue.empty():
            frame = self.frame_queue.get()
            # Some conditioning of the camera image
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mirrored_frame = cv2.flip(rotated_frame, 0)
            self.gray_frame = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)

            # Display the grayscale frame
            self.view.setImage(self.gray_frame, autoRange=self.view.update_range,
                                            autoLevels=self.view.update_levels)
            # Update ROIs
            self.on_view_roi_selected(self.view.roi)
            self.update_gaussian_fit()

            self.mirrorLabel.setText(f"{self.mirror.position:.3f}")
            self.feedback = self.feedback = self.source2_mean * self.feedback_K + self.feedback_offset
            self.feedbackLabel.setText(f"{self.feedback:.3f}")

            if self.regulator_on:
                current_position = self.mirror.position
                self.pid.update(self.feedback)
                correction = self.pid.getOutput()
                new_position = current_position + correction
                print(f"Setpoint: {self.pid.setpoint}, Feedback: {self.feedback}, Output: {correction}, New Pos: {new_position}")
                if abs(new_position - self.mirror_init_pos) <= self.mirror_envelop / 2:
                    self.mirror.position = new_position
                    while self.mirror.state() == tango.DevState.MOVING:
                        print("Motor is in motion!")
                        time.sleep(0.3)
                else:
                    print("Correction is outside of the envelop")

                # Wait for next calculation
                time.sleep(self.pid.sample_time)
        # self.clear_buffer(self.update_rate * 2)


if __name__ == '__main__':

    if len(sys.argv) > 2:
        stream_url = sys.argv[1]
        mirror_motor = sys.argv[2]
    else:
        print("Not enough arguments provided. Using the default URL: http://b-softimax-rpi-1:5000/stream")
        stream_url = "http://b-softimax-rpi-1:5000/stream"
        mirror_motor = "a_m3_pitch"

    cbmr_app = MainWidget(stream_url, mirror_motor)
    cbmr_app.setWindowTitle("Camera Based Mirror Regulator")
    cbmr_app.resize(1200, 1000)
    cbmr_app.show()

    sys.exit(app.exec_())
