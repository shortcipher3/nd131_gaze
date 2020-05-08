'''
This is a class for the facial landmarks detection model. If using a different model from the
default you may want to revisit some of the assumptions.

This model can be run independently with the default parameters as:
python3 src/facial_landmarks_detection.py

python3 src/facial_landmarks_detection.py --input=data/image_100_face.png --landmarks=models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009 --device=CPU --log-level=info

To modify the parameters check the help:
python3 src/facial_landmarks_detection.py --help
'''
import time
import logging
from typing import Dict, Any
import numpy as np
import cv2
from openvino.inference_engine import IECore


class FacialLandmarksDetector:
    '''
    Class for the Facial Landmark Detector Model.
    '''
    def __init__(self, model_name: str, device: str='CPU', input_width=48, input_height=48):
        self.model = model_name
        self.device = device
        self.input_width = input_width
        self.input_height = input_height
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

    def preprocess_input(self, image: np.ndarray, preserve_aspect_ratio: bool=False):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        Default model takes an input of a face crop 1x3x48x48 bgr BxCxHxW

        Parameters
        ----------
            image: image to run preprocessing on, should be a cropped face image
            preserve_aspect_ratio: boolean, https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html specifies for different models

        Returns
        -------
            batch: with preprocessing applied
            normalization_consts: ratio of the image pixels to image with padding
        """
        normalization_consts = [1.0, 1.0]
        if preserve_aspect_ratio:
            rows, cols, _ = image.shape
            fx = self.input_height * 1.0 / cols
            fy = self.input_width * 1.0 / rows
            if fx < fy:
                fy = fx
            else:
                fx = fy
            resized = cv2.resize(image.copy(), (0, 0), fx=fx, fy=fy)
            batch = np.zeros((self.input_height, self.input_width, 3), np.uint8)
            normalization_consts = [resized.shape[1] * 1.0 / batch.shape[1],
                                    resized.shape[0] * 1.0 / batch.shape[0]]
            batch[:resized.shape[0], :resized.shape[1], :] = resized
        else:
            batch = cv2.resize(image.copy(), (self.input_width, self.input_height))
        batch = batch.transpose((2,0,1)) # Channels first
        return batch[np.newaxis, :, :, :], normalization_consts

    def preprocess_output(self, raw_detections: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Change the format of the detections for use by the next model.

        Default model produces output landmarks 1x10 (x0, y0, ... x4, y4)

        Parameters
        ----------
            raw_detections: the facial landmarks estimation network output as produced by OpenVINO, an array of detections

        Returns
        -------
            detections: a list of detections meeting the criteria
        """
        logging.info(f'raw detections {raw_detections}')
        logging.info(f'raw detections {raw_detections.keys()}')
        for key, value in raw_detections.items():
            arr = np.squeeze(value)
            return {'left_eye': {'x': arr[0].item(), 'y': arr[1].item()},
                    'right_eye': {'x': arr[2].item(), 'y': arr[3].item()},
                    'nose': {'x': arr[4].item(), 'y': arr[5].item()},
                    'left_mouth': {'x': arr[6].item(), 'y': arr[7].item()},
                    'right_mouth': {'x': arr[8].item(), 'y': arr[9].item()}
                   }
        return {}

    def visualize_detections(self, image: np.ndarray, detections: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        visualize detected landmarks on the provided image

        Parameters
        ----------
            image: the image to put the landmarks on
            detections: the processed landmarks

        Returns
        -------
            visualization_img: the image with the landmarks
        """
        img = image.copy()
        cv2.circle(img,
                   (int(detections['left_eye']['x']*image.shape[1]), int(detections['left_eye']['y']*image.shape[0])),
                   5, [255, 0, 0], -1)
        cv2.circle(img,
                   (int(detections['right_eye']['x']*image.shape[1]), int(detections['right_eye']['y']*image.shape[0])),
                   5, [0, 0, 255], -1)
        cv2.circle(img,
                   (int(detections['nose']['x']*image.shape[1]), int(detections['nose']['y']*image.shape[0])),
                   5, [0, 255, 0], -1)
        cv2.circle(img,
                   (int(detections['left_mouth']['x']*image.shape[1]), int(detections['left_mouth']['y']*image.shape[0])),
                   5, [255, 125, 0], -1)
        cv2.circle(img,
                   (int(detections['right_mouth']['x']*image.shape[1]), int(detections['right_mouth']['y']*image.shape[0])),
                   5, [0, 75, 255], -1)
        return img

    def convert_to_full_frame_coordinates(self, detections: Dict[str, Dict[str, float]], face_coordinates: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        def adjust_coord(coord, face_min, length):
            return face_min + coord * length
        out = {'left_eye': {'x': adjust_coord(detections['left_eye']['x'], face_coordinates['x_min'], face_coordinates['width']),
                            'y': adjust_coord(detections['left_eye']['y'], face_coordinates['y_min'], face_coordinates['height'])},
               'right_eye': {'x': adjust_coord(detections['right_eye']['x'], face_coordinates['x_min'], face_coordinates['width']),
                             'y': adjust_coord(detections['right_eye']['y'], face_coordinates['y_min'], face_coordinates['height'])},
               'nose': {'x': adjust_coord(detections['nose']['x'], face_coordinates['x_min'], face_coordinates['width']),
                        'y': adjust_coord(detections['nose']['y'], face_coordinates['y_min'], face_coordinates['height'])},
               'left_mouth': {'x': adjust_coord(detections['left_mouth']['x'], face_coordinates['x_min'], face_coordinates['width']),
                              'y': adjust_coord(detections['left_mouth']['y'], face_coordinates['y_min'], face_coordinates['height'])},
               'right_mouth': {'x': adjust_coord(detections['right_mouth']['x'], face_coordinates['x_min'], face_coordinates['width']),
                               'y': adjust_coord(detections['right_mouth']['y'], face_coordinates['y_min'], face_coordinates['height'])},
        }
        return out

if __name__ == '__main__':
    # parse input arguments
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Estimate facial landmarks in a face image')
    parser.add_argument('--input',
                        default='data/image_100_face.png',
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
    print(detections)
    with open(f'{args.output}/face_landmarks.json', 'w') as f:
        json.dump(detections, f)


