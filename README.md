# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

python3 downloader.py --name face-detection-adas-binary-0001 --precisions=INT8,FP16,FP32,FP32-INT1 -o /workspace/models/
python3 downloader.py --name head-pose-estimation-adas-0001 --precisions=INT8,FP16,FP32 -o /workspace/models/
python3 downloader.py --name landmarks-regression-retail-0009 --precisions=INT8,FP16,FP32 -o /workspace/models/
python3 downloader.py --name gaze-estimation-adas-0002 --precisions=INT8,FP16,FP32 -o /workspace/models/

## Demo
*TODO:* Explain how to run a basic demo of your model.

python3 src/face_detection.py --input=data/image_100.png --detection=models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --device=CPU --log-level=info

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

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

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.


