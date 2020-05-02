'''
This is a class for the facial landmarks detection model. If using a different model from the
default you may want to revisit some of the assumptions.

This model can be run independently with the default parameters as:
python3 src/facial_landmarks_detection.py

To modify the parameters check the help:
python3 src/facial_landmarks_detection.py --help
'''
import time
import logging
from typing import Any, Dict, List
import numpy as np
import cv2
from openvino.inference_engine import IECore


class FacialLandmarksDetector:
    '''
    Class for the Facial Landmark Detector Model.
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
        logging.info(f'Time taken to load facial landmarks detection model = {(t2-t1)/cv2.getTickFrequency()} seconds')

        ### Check model ###
        # Get the supported layers of the network
        supported_layers = core.query_network(network=self.net, device_name=self.device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found in facial landmarks detections model: {}".format(unsupported_layers))
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
        Wait for an asynchronous call to finish and return the detections

        Returns
        -------
            raw_detections: the network's detections
        """
        while True:
            status = infer_request_handle.wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        raw_detections = infer_request_handle.outputs
        return raw_detections

    def sync_detect(self, batch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Synchronously run prediction on a batch with the network

        Parameters
        ----------
            batch: the batch of images to perform inference on

        Returns
        -------
            detections_arr: the array of detections
        """
        t1 = cv2.getTickCount()
        detections = self.exec_net.infer({self.input_blob: batch})
        t2 = cv2.getTickCount()
        logging.info(f'Time taken to execute facial landmarks detection model = {(t2-t1)/cv2.getTickFrequency()} seconds')
        return detections

    def preprocess_input(self, image: np.ndarray, width: int=672, height: int=384, preserve_aspect_ratio: bool=False):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        Default model takes an input of a face crop 1x3x48x48 bgr BxCxHxW

        Parameters
        ----------
            image: image to run preprocessing on
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

    def preprocess_output(self, raw_detections: np.ndarray, threshold: float=0.3, whitelist_filter: List[int]=[1], normalization_consts: List[float]=[1.0, 1.0]) -> List[Dict[str, Any]]:
        """
        Change the format of the detections for use by the next model.

        Default model produces output landmarks 1x10 (x0, y0, ... x4, y4)

        Parameters
        ----------
            raw_detections: the facial landmarks estimation network output as produced by OpenVINO, an array of detections
            threshold: discard detections with a score lower than this threshold
            whitelist_filter: the class ids to include, if empty it includes all of them

        Returns
        -------
            detections: a list of detections meeting the criteria
        """
        logging.info(f'raw detections {raw_detections}')
        arr = np.concatenate(raw_detections['detection_out'][:, 0, :, :], axis=0)
        # filter based on threshold
        arr = arr[arr[:, 2]>threshold, :]
        #logging.info(arr) #TODO bbox output looks wrong for batch size > 1
        #logging.info(arr.shape)
        if whitelist_filter:
            arr = arr[np.isin(arr[:, 1], whitelist_filter), :]
        num_detections = arr.shape[0]
        detections = []
        for k in range(num_detections):
            x_min = arr[k, 3] / normalization_consts[0]
            x_max = arr[k, 5] / normalization_consts[0]
            y_min = arr[k, 4] / normalization_consts[1]
            y_max = arr[k, 6] / normalization_consts[1]
            width = np.abs(x_max - x_min)
            height = np.abs(y_max - y_min)
            area = width * height
            detections.append({'image_id': arr[k, 0],
                               'label': arr[k, 1],
                               'conf': arr[k, 2],
                               'x_min': x_min,
                               'y_min': y_min,
                               'x_max': x_max,
                               'y_max': y_max,
                               'width': width,
                               'height': height,
                               'area': area})
        logging.info(f'detections {detections}')
        return detections

    def visualize_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        img = image.copy()
        for det in detections:
            x_min = int(det['x_min'] * img.shape[1])
            x_max = int(det['x_max'] * img.shape[1])
            y_min = int(det['y_min'] * img.shape[0])
            y_max = int(det['y_max'] * img.shape[0])
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (125, 255, 51), thickness=2)
            cv2.putText(img,
                     f'score: {det["conf"]:.2f} label: {det["label"]}',
                     (x_min, y_min),
                     cv2.FONT_HERSHEY_SIMPLEX,
                     .5,
                     (0,0,255),
                     2,
                     cv2.LINE_AA)
        return img

if __name__ == '__main__':
    # parse input arguments
    import argparse
    parser = argparse.ArgumentParser(description='Estimate facial landmarks in a face image')
    parser.add_argument('--input',
                        default='data/image_100.png',
                        type=str,
                        help='open video file or image file sequence or a capturing device or an IP video stream for video capturing')
    parser.add_argument('--landmarks',
                        default='models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009',
                        type=str,
                        help='path to the facial landmarks detection model')
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

    # run the detection network and save the output
    image = cv2.imread(args.input)
    flm_det = FacialLandmarksDetector(args.landmarks, args.device)
    batch, _ = flm_det.preprocess_input(image)
    dets = flm_det.sync_detect(batch)
    detections = flm_det.preprocess_output(dets)
    img = flm_det.visualize_detections(image, detections)
    cv2.imwrite(f'{args.output}/face_landmarks.png', img)


