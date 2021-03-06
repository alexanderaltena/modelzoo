# MobileNet Setup

Basic setup

python 3.5 - 3.8

Step 1: Add and create virtual env for tflite
```
mkdir -p Projects/Python/tflite
cd Projects/Python/tflite
python -m pip install virtualenv
python -m venv tflite-env
```

Step 2: Activate env
```
source tflite-env/bin/activate
```

Step 3: (make sure you are in a venv) Install libraries for imageprocessing
```
sudo apt -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
```
Step 4: GUI functionallity
```
sudo apt -y install qt4-dev-tools libatlas-base-dev libhdf5-103 
```

Step 5: install openCV (if you have issues opencv-python==3.4.11.41 )
```
python -m pip install opencv-contrib-python==4.1.0.25
```

Step 6: Check out your processor and python version:
```
uname -m
python --version
```

step 6: Install TF
Open an Internet browser on your Pi and head to https://github.com/google-coral/pycoral/releases/. Scroll down to the list of wheel (.whl) files and find the group that matches your OS/processor, which should be “Linux (ARM 32).” In that group, find the link that corresponds to your version of Python (3.7 for me). Right-click and select 
Back in the terminal, enter the following:
```
python -m pip install <paste in .whl link>
```

Step 7: Run the webcam object detection script
```
python detect.py --modeldir '{path}/mobilenet'
```

* Based on:
https://www.digikey.com/en/maker/projects/how-to-perform-object-detection-with-tensorflow-lite-on-raspberry-pi/b929e1519c7c43d5b2c6f89984883588
