import numpy as np
import cv2

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
    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        choices=['CPU', 'GPU', 'MYRIAD', 'FPGA'],
                        help='the device to run inference on, one of CPU, GPU, MYRIAD, FPGA')
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
    parser.add_argument('--visualize',
                        default=False,
                        action='store_true',
                        help='visualize the intermediate model output')
    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output directory to save results')

    args = parser.parse_args()

    mc = MouseController('medium', 'medium')

    face_det = FaceDetector(args.detection, args.device)
    flm_det = FacialLandmarksDetector(args.landmarks, args.device)
    head_pose_est = HeadPoseEstimator(args.pose, args.device)
    gaze_est = GazeEstimator(args.gaze, args.device)

    inp = InputFeeder(args.input)

    if args.visualize:
        vw = cv2.VideoWriter(args.output + '/debug.mp4', cv2.VideoWriter_fourcc(*'avc1'), inp.get_fps(), inp.get_shape(), True)

    for frame in inp:
        fd_batch, _ = face_det.preprocess_input(frame)
        face_dets = face_det.sync_detect(fd_batch)
        face_detections = face_det.preprocess_output(face_dets)
        # there could be multiple faces detected, choose the largest
        largest_face = np.array((0, 0))
        face_detection = None
        for k, crop in enumerate(face_det.crop_face(frame, face_detections)):
            if crop.size > largest_face.size:
                largest_face = crop
                face_detection = face_detections[k]
        if not face_detection:
            if args.visualize:
                vw.write(frame)
            continue

        flm_batch, _ = flm_det.preprocess_input(largest_face)
        flm_dets = flm_det.sync_detect(flm_batch)
        flm_detections = flm_det.preprocess_output(flm_dets)

        pose_batch, _ = head_pose_est.preprocess_input(largest_face)
        pose_ests = head_pose_est.sync_detect(pose_batch)
        pose_estimations = head_pose_est.preprocess_output(pose_ests)

        gaze_inputs = gaze_est.preprocess_input(largest_face, flm_detections, pose_estimations)
        gaze_dets = gaze_est.sync_detect(gaze_inputs)
        gaze_vec = gaze_est.preprocess_output(gaze_dets)

        if args.visualize:
            visualize = frame.copy()
            visualize = face_det.visualize_detections(visualize, face_detections)
            flm_detections = flm_det.convert_to_full_frame_coordinates(flm_detections, face_detection)
            visualize = flm_det.visualize_detections(visualize, flm_detections)
            visualize = head_pose_est.visualize_estimations(visualize, pose_estimations)
            visualize = gaze_est.visualize_gaze(visualize, gaze_vec, flm_detections)
            vw.write(visualize)
    if args.visualize:
        vw.release()

