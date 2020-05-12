import numpy as np
import cv2

from face_detection import FaceDetector
from facial_landmarks_detection import FacialLandmarksDetector
from input_feeder import InputFeeder

if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser(description='Use computer vision to control your mouse position with your gaze')
    parser.add_argument('--input',
                        default='data/demo.mp4',
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
    parser.add_argument('--log-level',
                        default='error',
                        type=str,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the log level, one of debug, info, warning, error, critical')

    args = parser.parse_args()

    # set log level
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    log_level = LEVELS.get(args.log_level, logging.ERROR)
    logging.basicConfig(level=log_level)

    face_det = FaceDetector(args.detection, args.device)
    flm_det = FacialLandmarksDetector(args.landmarks, args.device)

    inp = InputFeeder(args.input)

    output_path = 'facial_landmarks.mp4'
    #vw = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    vw = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), 15, (1664, 1664), True)
    for frame in inp:
        batch, _ = face_det.preprocess_input(frame)
        face_dets = face_det.sync_detect(batch)
        face_detections = face_det.preprocess_output(face_dets)
        # there could be multiple faces detected, choose the largest
        largest_face = np.array((0, 0))
        face_detection = None
        for k, crop in enumerate(face_det.crop_face(frame, face_detections)):
            if crop.size > largest_face.size:
                largest_face = crop
                face_detection = face_detections[k]
        if not face_detection:
            vw.write(frame)
            continue

        flm_batch, _ = flm_det.preprocess_input(largest_face)
        flm_dets = flm_det.sync_detect(flm_batch)
        flm_detections = flm_det.preprocess_output(flm_dets)

        visualization = face_det.visualize_detections(frame, face_detections)
        flm_img_dets = flm_det.convert_to_full_frame_coordinates(flm_detections, face_detection)
        visualization = flm_det.visualize_detections(visualization, flm_img_dets)
        vw.write(visualization)
        #cv2.imwrite('visualization.png', visualization)
        #break
    vw.release()


