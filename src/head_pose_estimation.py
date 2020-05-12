'''
This is a class for the head pose estimation model. If using a different model from the default you may
want to revisit some of the assumptions.

This model can be run independently with the default parameters as:
python3 src/head_pose_estimation.py

python3 src/head_pose_estimation.py --input=data/image_100_face.png --pose=models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001 --device=CPU --log-level=info

To modify the parameters check the help:
python3 src/head_pose_estimation.py --help
'''
import time
import logging
from typing import Dict
import numpy as np
import cv2
from openvino.inference_engine import IECore


class HeadPoseEstimator:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name: str, device: str='CPU'):
        self.model = model_name
        self.device = device
        model_weights = self.model + '.bin'
        model_structure = self.model + '.xml'

        ### Load model ###
        t1 = cv2.getTickCount()
        core = IECore()
        self.net = core.read_network(model=model_structure, weights=model_weights)
        self.exec_net = core.load_network(network=self.net, device_name=self.device)
        t2 = cv2.getTickCount()
        logging.info(f'Time taken to load head pose estimation model = {(t2-t1)/cv2.getTickFrequency()} seconds')

        ### Check model ###
        # Get the supported layers of the network
        supported_layers = core.query_network(network=self.net, device_name=self.device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found in head pose estimation model: {}".format(unsupported_layers))
            logging.error("Check whether extensions are available to add to IECore.")
            exit(1)

        # Get the input layer
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))

    def async_detect(self, batch: np.ndarray):
        """
        Asynchronously run prediction on a batch with the network

        Parameters
        ----------
            batch: the batch of images to perform inference on

        Returns
        -------
            infer_request_handle: the handle for the asynchronous request, needed by async_wait
        """
        infer_request_handle = self.exec_net.start_async(request_id=0, inputs={self.input_blob: batch})
        return infer_request_handle

    def async_wait(self, infer_request_handle):
        """
        Wait for an asynchronous call to finish and return the estimations

        Returns
        -------
            raw_estimations: the network's estimations
        """
        while True:
            status = infer_request_handle.wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        raw_estimations = infer_request_handle.outputs
        return raw_estimations

    def sync_detect(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Synchronously run prediction on a batch with the network

        Parameters
        ----------
            batch: the batch of images to perform inference on

        Returns
        -------
            estimations_arr: the array of estimations
        """
        t1 = cv2.getTickCount()
        estimations = self.exec_net.infer({self.input_blob: batch})
        t2 = cv2.getTickCount()
        logging.info(f'Time taken to execute head pose estimation model = {(t2-t1)/cv2.getTickFrequency()} seconds')
        return estimations

    def preprocess_input(self, image: np.ndarray, width: int=60, height: int=60, preserve_aspect_ratio: bool=False):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        Default model takes an input of 1x3x60x60 BGR 1xCxHxW

        Parameters
        ----------
            image: image to run preprocessing on, should be a cropped face image
            width: desired width
            height: desired height
            preserve_aspect_ratio: boolean, https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html specifies for different models

        Returns
        -------
            batch: with preprocessing applied
            normalization_consts: ratio of the image pixels to image with padding
        """
        normalization_consts = [1.0, 1.0]
        if preserve_aspect_ratio:
            rows, cols, _ = image.shape
            fx = height * 1.0 / cols
            fy = width * 1.0 / rows
            if fx < fy:
                fy = fx
            else:
                fx = fy
            resized = cv2.resize(image.copy(), (0, 0), fx=fx, fy=fy)
            batch = np.zeros((height, width, 3), np.uint8)
            normalization_consts = [resized.shape[1] * 1.0 / batch.shape[1],
                                    resized.shape[0] * 1.0 / batch.shape[0]]
            batch[:resized.shape[0], :resized.shape[1], :] = resized
        else:
            batch = cv2.resize(image.copy(), (width, height))
        batch = batch.transpose((2,0,1)) # Channels first
        return batch[np.newaxis, :, :, :], normalization_consts

    def preprocess_output(self, raw_estimations: np.ndarray) -> Dict[str, float]:
        """
        Change the format of the estimations for use by the next model.

        Default model produces a diction with angles in degrees of shape [1, 1]
          * angle_y_fc (yaw)
          * angle_p_fc (pitch)
          * angle_r_fc (roll)

        Parameters
        ----------
            raw_estimations: the head pose estimation network output as produced by OpenVINO, an array of estimations

        Returns
        -------
            detections: a list of detections meeting the criteria
        """
        logging.info(f'raw estimations {raw_estimations}')
        estimations = {'yaw': np.squeeze(raw_estimations['angle_y_fc']).item(),
                       'pitch': np.squeeze(raw_estimations['angle_p_fc']).item(),
                       'roll': np.squeeze(raw_estimations['angle_r_fc']).item()}
        logging.info(f'processed estimations {estimations}')
        return estimations

    def visualize_estimations(self, image: np.ndarray, estimations: Dict[str, float]) -> np.ndarray:
        img = image.copy()
        roll = estimations['roll'] * np.pi / 180.0
        pitch = estimations['pitch'] * np.pi / 180.0
        yaw = estimations['yaw'] * np.pi / 180.0
        rvec = (yaw,   roll,  pitch)
        tvec = (0, 0, 1)
        points = np.array([[0,0,0],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]], np.float32)
        camera_matrix = np.array([[100, 0, image.shape[1]/2.0],
                                  [0, 100, image.shape[0]/2.0],
                                  [0,  0, 1]], np.float32)
        image_points, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, ())
        image_points = np.squeeze(image_points)
        logging.info(f'pose points {image_points}')
        img = cv2.line(img, tuple(image_points[0, :]), tuple(image_points[1, :]), (255, 0, 0), 5)
        img = cv2.line(img, tuple(image_points[0, :]), tuple(image_points[2, :]), (0, 255, 0), 5)
        img = cv2.line(img, tuple(image_points[0, :]), tuple(image_points[3, :]), (0, 0, 255), 5)
        return img

if __name__ == '__main__':
    # parse input arguments
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Estimate head pose given a cropped face image')
    parser.add_argument('--input',
                        default='data/image_100_face.png',
                        type=str,
                        help='open video file or image file sequence or a capturing device or an IP video stream for video capturing')
    parser.add_argument('--pose',
                        default='models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001',
                        type=str,
                        help='path to the head pose estimation model')
    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        choices=['CPU', 'GPU', 'MYRIAD', 'FPGA'],
                        help='the device to run inference on, one of CPU, GPU, MYRIAD, FPGA')
    parser.add_argument('--log-level',
                        default='error',
                        type=str,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the log level, one of debug, info, warning, error, critical')
    parser.add_argument('--output',
                        default='output',
                        type=str,
                        help='output directory to save results')

    args = parser.parse_args()

    # set log level
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    log_level = LEVELS.get(args.log_level, logging.ERROR)
    logging.basicConfig(level=log_level)

    # run the estimation network and save the output
    image = cv2.imread(args.input)
    head_pose_est = HeadPoseEstimator(args.pose, args.device)
    batch, _ = head_pose_est.preprocess_input(image)
    dets = head_pose_est.sync_detect(batch)
    estimations = head_pose_est.preprocess_output(dets)
    img = head_pose_est.visualize_estimations(image, estimations)
    cv2.imwrite(f'{args.output}/head_pose.png', img)
    with open(f'{args.output}/head_pose.json', 'w') as f:
        json.dump(estimations, f)

