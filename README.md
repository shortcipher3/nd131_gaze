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
python3 src/main.py
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

### Building the docker image
docker build -t shortcipher3/gaze -f nd131_gaze/docker/Dockerfile --build-arg DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/16612/l_openvino_toolkit_p_2020.2.120.tgz .

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
If you select a log-level of debug many of the inference steps are timed and the results will be
logged to the screen.  However, I chose to benchmark my project using the `%lprun` magic in
ipython. To run the profiling step simply start up an ipython3 instance and feed it the following
commands (these can also be put in a script).

`ipython3`
```
%load_ext line_profiler
import sys
sys.path.append('src')
from argparse import Namespace
import main
args = Namespace(input='data/demo.mp4',
    device='CPU',
    detection='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
    landmarks='models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009',
    pose='models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001',
    gaze='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002',
    visualize=False,
    output='output',
    log_level='error')
%lprun -T output/benchmark_FP32.txt -f main.main main.main(args)
```

The results of the benchmarks are saved in output/benchmark_FP32.txt and output/benchmark_FP16.txt,
I have summarized some of the results in the following table (note: the units are in microseconds).

model     | device | precision | loading | input processing | output processing | inference
----------|--------|-----------|---------|------------------|-------------------|-----------
detection | CPU    | FP32-INT1 | 534255  | 1756             | 2851              | 31365.9
landmarks | CPU    | FP32      | 193327  | 105.4            | 2011.7            |  1617.7
head pose | CPU    | FP32      | 206189  | 149.5            | 2302.5            |  3130.4
gaze      | CPU    | FP32      | 240494  | 408.2            |  989.9            |  3997.6
landmarks | CPU    | FP16      | 251583  | 107.1            | 2090.9            |  1560.5
head pose | CPU    | FP16      | 292552  | 152.8            | 2481.1            |  3410.1
gaze      | CPU    | FP16      | 360563  | 417.1            | 1057.9            |  4310.0


## Results
The results of the benchmarks are saved in output/benchmark_FP32.txt and output/benchmark_FP16.txt

The benchmarks give a pretty clear result - 95% of the time is spent moving the mouse and
displaying the images. Loading the models is a small fraction of the total time and hardly
impacts the results when processing many frames, however if only a couple of frames were proceessed
this cost would be significant. The input processing is about an order of magnitude shorter than the
inference. Some of the post processing steps are on par with the inference - I'm guessing this is
because I logged the outputs for debugging purposes, the output processing would be an easy place
to optimize.

I did not see INT8 models for these models available to download and for the detection model I only
found one model which was FP32 with some binary layers. The face detection model is the slowest
model by an order of magnitude, to speed the computer vision pipeline up one could simply run that
model asynchronously and then run the other models while waiting for it to finish.

Improvements in throughput might be seen with some of the other Intel hardware especially if larger
batch sizes were used.  This application seems more focused on latency so large batch sizes don't
make as much sense.

The purpose of using lower model precision is to trade-off accuracy (lowered) for inference time
(faster), however my experiments didn't see that trade-off.  I assume it is because these are light
models that aren't using the full CPU / memory capacity.  Some models I have worked with for other
projects are hundreds of MBs whereas these models are all less than 10 MB.  Perhaps with larger
models we would see more improvement from lowering model precision.

## Stand Out Suggestions
### Provided Stand Out Suggestions
* Can you improve your inference speed without significant drop in performance by changing the
precision of some of the models? In your README, include a short write-up that explains the
procedure and the experiments you ran to find out the best combination of precision.
* Benchmark the running times of different parts of the preprocessing and inference pipeline
and let the user specify a CLI argument if they want to see the benchmark timing. Use the
`get_perf_counts`  API to print the time it takes for each layer in the model.
* Use the VTune Amplifier to find hotspots in your Inference Engine Pipeline. Write a short
write-up in the README about the hotspots in your system and how you solved them.
* There will be certain edge cases that will cause your system to not function properly.
Examples of this include: lighting changes, multiple people in the same input frame, and so on.
Make changes in your preprocessing and inference pipeline to solve some of these issues. Write a
short write-up in the README about the problem it caused and your solution.
* Add a toggle to the UI to shut off the camera feed and show stats only (as well as to toggle the
camera feed back on). Show how this affects performance and power as a short write up in the
README file.
* Build an inference pipeline for both video file and webcam feed as input. Allow the user to
select their input option in the command line arguments.

### Achieved Stand Out Suggestions

* I built my project with docker, python3, and the latest OpenVINO. I included features such as
f-strings and type hints.
* Build an inference pipeline for both video file and webcam feed as input. Allow the user to
select their input option in the command line arguments.
My project handles this, see the documentation on the InputFeeder
* Add a toggle to the UI to shut off the camera feed and show stats only (as well as to toggle the
camera feed back on). Show how this affects performance and power as a short write up in the
README file.
Rather than creating a toggle I profiled the whole application and showed that displaying the
camera feed consumes about ~2% of the total time in the application (mouse movement consumes
~93%).  So removing the display saves a significant amount of resources relative to the computer
vision pipeline, but is insignificant when compared with the mouse controller.
* There will be certain edge cases that will cause your system to not function properly.
Examples of this include: lighting changes, multiple people in the same input frame, and so on.
Make changes in your preprocessing and inference pipeline to solve some of these issues. Write a
short write-up in the README about the problem it caused and your solution.
The edge cases I identified and built in some robustness too are described in the Edge Cases
section of this document.
* Can you improve your inference speed without significant drop in performance by changing the
precision of some of the models? In your README, include a short write-up that explains the
procedure and the experiments you ran to find out the best combination of precision.
I explained my experiments and results in the benchmarking section. I also make proposals for how
best to speed up the computer vision pipeline under the async inference section as well as the
benchmarking section.

### Async Inference
I did not test async inference, due to time constraints. I also don't see a lot of value when the
mouse controller and visualization is taking 95% of the application's time.  However, to optimize
the computer vision pipeline it would clearly be beneficial to run async inference on face
detection, and run the other models on the output of the last face detection before checking for
the next output of the face detection model.

I also think it would be neat to use an Actor Model to optimize the computer vision pipeline, such
as Thespian - I may revisit this.

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
