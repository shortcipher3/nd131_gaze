# Computer Pointer Controller

The idea behind this project was to control the mouse using gaze by creating a computer vision
pipeline that detects the face, given the face detects the head pose and the eye locations, then
uses that information to estimate the gaze direction of the eyes and move the mouse based on the
gaze direction.

## Project Set Up and Installation

There are two ways to run my project, with docker and without docker.

### Docker setup
Prerequisites
* docker installed
* computer supports X11 (can use XQuartz on Mac, out of the box support on Linux)
Steps:
1. Set up X11 forwarding and run
```
xhost +
docker run --rm -ti -v ~/.Xauthority:/root/.Xauthority:rw --env "DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --net=host shortcipher3/gaze
```

### Dockerless setup
Prerequisites
* OpenVINO 2020.2 installed
* python3 installed
* graphics supported
* python3 virtual environment setup
Steps:
1. clone the repo
```
git clone git@github.com:shortcipher3/nd131_gaze.git
cd nd131_gaze
```
2. download the models
```
python3 downloader.py --name face-detection-adas-binary-0001 --precisions=INT8,FP16,FP32,FP32-INT1 -o /models/
python3 downloader.py --name head-pose-estimation-adas-0001 --precisions=INT8,FP16,FP32 -o /models/
python3 downloader.py --name landmarks-regression-retail-0009 --precisions=INT8,FP16,FP32 -o /models/
python3 downloader.py --name gaze-estimation-adas-0002 --precisions=INT8,FP16,FP32 -o /models/
```
3. install the requirements
```
virtualenv -p $(which python3) python_env
source python_env/bin/activate
pip install -r requirements.txt
```
4. run
```
python3 src/main.py
```

### Directory Structure
The model files are stored in the models directory.

Test data is stored in the data directory.

Output data is stored in the output directory by default.

Source code is stored in the src directory.

## Demo

To run a basic demo of my model run:
```
python3 src/main.py
```

To run with arguments you can get a list of arguements and their descriptions by running:
```
python3 src/main.py --help
```

The documentation section also includes more details.

## Documentation

There are six main classes implemented in this project:
* MouseController (in mouse_controller.py - mostly provided, minor modifications)
* InputFeeder (in input_feeder.py - a thin wrapper around `cv2.VideoCapture`)
* FaceDetector (in face_detection.py - model class for detecting faces in an image)
* FacialLandmarksDetector (in facial_landmarks_detection.py - model for finding eyes, nose, and
mouth keypoints in a face image)
* HeadPoseEstimator (in head_pose_estimation.py - model for finding head pose in a face image)
* GazeEstimator (in gaze_estimation.py - model for finding the gaze direction given a face image
and the face pose and eye positions)

I will now show how you can test each of the modules I implemented.

### InputFeeder
I treated InputFeeder as a thin wrapper for `cv2.VideoCapture`, which can read an image, a video
file, a video camera, a gstreamer pipeline, etc.  Running the file by itself will extract the
specified number of frames from the video/image and write them as images to the output folder.

The requirements specified that it should support still images, in which case it should repeatedly
produce the same image, to accomplish this I count the number of frames I get from the video source
if I only get one image I consider it a still image mode and continue to produce the same output
image. An alternative I considered was to use the `VideoCapture.get(cv2.CAP_PROP_FRAME_COUNT)`, but
I found this wasn't a reliable indicator.

Some example usage:

Extract 10 frames from a video file
`python3 src/input_feeder.py --input data/demo.mp4 --frames 10 --output output`

Save the image as 30 frames
`python3 src/input_feeder.py --input data/image_100.png --frames 30 --output .`

Extract 100 frames from the live video camera
`python3 src/input_feeder.py --input 0 --frames 100 --output output`

### FaceDetector
The face detector test detects the faces in one image and draws bounding boxes around the faces,
it also crops out the faces and saves them to an output folder.

You can specify where the detection model is stored, the device to run on, the output folder, and
a log level.

Example usage:
```
python3 src/face_detection.py --input=data/image_100.png \
  --detection=models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
  --device=CPU --log-level=debug --output output
```

### FacialLandmarksDetector
The facial landmarks detector test detects the left eye, right eye, nose, left mouth corner, and
right mouth corner in a cropped face image.  It draws these landmarks on the face and writes the
image out to a file it also writes the coordinates out to a json file.

You can specify where the detection model is stored, the device to run on, the output folder, and
a log level.

Example usage:
```
python3 src/facial_landmarks_detection.py --input=data/image_100_face.png \
  --landmarks=models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
  --device=CPU --log-level=debug --output output
```

### HeadPoseEstimator
The head pose estimator test detects the pose of the head given a cropped face image, it draws an
axis to the head image that represents the head pose and writes that image to the output folder, it
also writes the output to a json file.

You can specify where the estimation model is stored, the device to run on, the output folder, and
a log level.

Example usage:
```
python3 src/head_pose_estimation.py --input=data/image_100_face.png \
  --pose=models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
  --device=CPU --log-level=debug --output output
```

### GazeEstimator
The gaze estimator test takes a face crop image, the head pose in json format, and the facial
landmarks in json format and estimates the gaze direction.  It draws the gaze direction onto the
image and writes the resulting visualization to the output folder.

You can specify where the detection model is stored, the device to run on, the output folder, and
a log level.

Example usage:
```
python3 src/gaze_estimation.py --input=data/image_100_face.png \
  --input-landmarks data/image_100_face_landmarks.json \
  --input-pose data/image_100_head_pose.json \
  --gaze=models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 \
  --device=CPU --log-level=debug --output output
```

### Running the main program
The main program combines all of these modules into one application that detects the estimates a
person's gaze from a video file, live video, or image and based on the gaze direction moves the
mouse.

You can specify:
* --input - the input 0 for v4l2 live video0, filename for image or video file
* --device - the device to run on, one of ['CPU', 'GPU', 'MYRIAD', 'FPGA']
* --detection - the path to the face detection model
* --landmarks - the path to the facial landmarks detection model
* --pose - the path to the head pose model
* --gaze - the path to the gaze model
* --visualize - if present then the individual model visualizations are displayed and saved to an
output file
* --output - the path to the output folder
* --log-level - the log level, one of ['debug', 'info', 'warning', 'error', 'critical']

Example usage:
```
python3 src/main.py --input data/demo.mp4 --device CPU \
  --detection models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 \
  --landmarks models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 \
  --pose models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 \
  --gaze models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002 \
  --visualize \
  --log-level error \
  --output output
```

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

model | loading | input processing | output processing | inference


## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
### Provided Stand Out Suggestions
* Can you improve your inference speed without significant drop in performance by changing the precision of some of the models? In your README, include a short write-up that explains the procedure and the experiments you ran to find out the best combination of precision.
* Benchmark the running times of different parts of the preprocessing and inference pipeline and let the user specify a CLI argument if they want to see the benchmark timing. Use the `get_perf_counts`  API to print the time it takes for each layer in the model.
* Use the VTune Amplifier to find hotspots in your Inference Engine Pipeline. Write a short write-up in the README about the hotspots in your system and how you solved them.
* There will be certain edge cases that will cause your system to not function properly. Examples of this include: lighting changes, multiple people in the same input frame, and so on. Make changes in your preprocessing and inference pipeline to solve some of these issues. Write a short write-up in the README about the problem it caused and your solution.
* Add a toggle to the UI to shut off the camera feed and show stats only (as well as to toggle the camera feed back on). Show how this affects performance and power as a short write up in the README file.
* Build an inference pipeline for both video file and webcam feed as input. Allow the user to select their input option in the command line arguments.

### Achieved Stand Out Suggestions
I built my project with docker, python3, and the latest OpenVINO.  I included features such as f-strings and type hints.

### Async Inference
I did not test async inference, due to time constraints.

### Edge Cases
Some of the edge cases I encountered and the solution I provided include:
* when no face is detected - I display the image without moving the mouse, also no detection
information is displayed on the frame if the --visualize flag is set
* when multiple faces are detected - I sort through the faces and select the largest one, I
assume the person closest to the camera is the one trying to control the mouse
* the eye can be located close to the edge of the picture or the face crop, in which case there may
not be 30 pixels on each side of the eye for when the eyes are cropped out for gaze estimation - in
this case I pad the image by reflecting the image across the border
* when the mouse reaches the corner of the screen an error is raised and the demo exits - I changed
pyautogui to failsafe mode so the other parts of the demo will continue to run
