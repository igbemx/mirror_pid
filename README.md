# Camera Based Mirror Regulator (CBMR)

This project implements a camera-based mirror regulator application using OpenCV, PyQt5, and Tango libraries. The application aims to regulate the position of a mirror based on the feedback received from a camera image analysis.

## Features

- Camera capture: Streams video frames from a specified URL.
- Image processing: Converts frames to grayscale and performs necessary image processing tasks.
- Region of Interest (ROI) selection: Allows users to define ROIs for source 1 and source 2 analysis.
- Gaussian fit: Fits a Gaussian curve to the selected ROI for source 1 to determine its mean value.
- PID controller: Implements a Proportional-Integral-Derivative (PID) controller to regulate the mirror position based on the feedback signal.
- Tango device interaction: Communicates with a Tango device (mirror motor) to adjust its position. (Optional)
- Real-time feedback: Displays the current mirror position, feedback signal, and source 1 and source 2 mean values.
- Control panel: Provides a user interface for controlling various aspects of the application, including:
    - Start/stop acquisition
    - Start/stop regulation
    - Update rate
    - Mirror setpoint
    - PID controller gains
    - Feedback calibration (must be used carefully)

## Requirements

- Python 3.x
- OpenCV
- PyQt5
- Tango Control System (for mirror control)
- NumPy
- SciPy

## Getting Started

1. Clone this repository.
2. Set TANGO_HOST if not set up yet using `export TANGO_HOST="<database_host_name>:10000"`.
3. Make a conda environment and install the required libraries using `conda env create -f mirror_pid_conda.yml`.
4. Run the application using `python mirror_pid.py <camera_stream_url> <mirror_motor_alias>`.
     - If no arguments are provided, the application will use the default camera stream URL "http://b-softimax-rpi-1:5000/stream" and mirror motor alias "a_m3_pitch".
