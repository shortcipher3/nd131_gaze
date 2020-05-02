from face_detection import FaceDetector
from facial_landmarks_detection import FacialLandmarksDetector
from head_pose_estimation import HeadPoseEstimator
from gaze_estimation import GazeEstimator
from mouse_controller import MouseController
from input_feeder import InputFeeder

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Use computer vision to control your mouse position with your gaze')
    parser.add_argument('--input',
                        default='bin/demo.mp4',
                        type=str,
                        help='open video file or image file sequence or a capturing device or an IP video stream for video capturing')
    parser.add_argument('--detection',
                        default='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
                        type=str,
                        help='path to the face detection model')
    parser.add_argument('--landmarks',
                        default='models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009',
                        type=str,
                        help='path to the facial landmarks model')
    parser.add_argument('--pose',
                        default='models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001',
                        type=str,
                        help='path to the head pose estimation model')
    parser.add_argument('--gaze',
                        default='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002',
                        type=str,
                        help='path to the gaze estimation model')

    args = parser.parse_args()

    mc = MouseController('medium', 'medium')


