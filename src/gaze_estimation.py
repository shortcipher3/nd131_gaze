'''
This is a class for the gaze estimation model. If using a different model from the default you may
want to revisit some of the assumptions.

This model can be run independently with the default parameters as:
python3 src/face_detection.py

To modify the parameters check the help:
python3 src/face_detection.py --help
'''
import time
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2
from openvino.inference_engine import IECore


class GazeEstimator:
    '''
    Class for the Gaze Estimation Model.
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
        logging.info(f'Time taken to load the gaze estimation model = {(t2-t1)/cv2.getTickFrequency()} seconds')

        ### Check model ###
        # Get the supported layers of the network
        supported_layers = core.query_network(network=self.net, device_name=self.device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found in gaze estimation model: {}".format(unsupported_layers))
            logging.error("Check whether extensions are available to add to IECore.")
            exit(1)

    def async_detect(self, inputs: Dict[str, Any]):
        """
        Asynchronously run prediction on a batch with the network

        Parameters
        ----------
            batch: the batch of images to perform inference on

        Returns
        -------
            infer_request_handle: the handle for the asynchronous request, needed by async_wait
        """
        infer_request_handle = self.exec_net.start_async(request_id=0, inputs=inputs)
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

    def sync_detect(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
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
        detections = self.exec_net.infer(inputs)
        t2 = cv2.getTickCount()
        logging.info(f'Time taken to execute gaze estimation model = {(t2-t1)/cv2.getTickFrequency()} seconds')
        return detections

    def crop_eye(self, image: np.ndarray, center: Tuple[float, float], width: int=60, height: int=60):
        """
        crops the eye based on eye center location, since we don't have information about the size
        of the eye which would be impacted by resolution of the image and distance from the camera,
        we are crossing our fingers and hoping it works

        Parameters
        ----------
            image: high resolution image
            center: the (x, y) coordinates of the eye
            width: the desired width of the cropped eye
            height: the desired height of the cropped eye

        Returns
        -------
            crop: the cropped eye
        """
        cv2.imwrite('pre_eye_crop.png', image)
        h = int(height/2)
        w = int(width/2)
        # pad so can crop without worrying about being too close to edges
        padded = cv2.copyMakeBorder(image, h, h, w, w, cv2.BORDER_REFLECT_101)
        try:
            logging.info(f'cropping eye from {int(center[1]-height/2)} to {int(center[1]+height/2)}' + \
                         f' and {int(center[0]-width/2)} to {int(center[0]+width/2)}')
            logging.info(f'cropping padded eye from {int(center[1])} to {int(center[1]+height)}' + \
                         f' and {int(center[0])} to {int(center[0]+width)}')
            crop = padded[int(center[1]):int(center[1] + height),
                         int(center[0]):int(center[0] + width),
                         :]
        except:
            raise ValueError('One or more of the inputs were not supported, perhaps the eyes are too close to the edge of the image or the input image was too small')
        if crop.shape[0] != height or crop.shape[1] != width:
            raise ValueError(f'The crop was the wrong size {crop.shape} desired {height}x{width}')
        cv2.imwrite('eye_crop.png', crop)
        crop = crop.transpose((2,0,1)) # Channels first
        return crop[np.newaxis, :, :, :]

    def preprocess_input(self, image: np.ndarray, landmarks: Dict[str, Dict[str, float]], pose: Dict[str, float]) -> Dict[str, Any]:
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.

        Default model takes an input of input left eye crop 1x3x60x60, right eye crop 1x3x60x60, head direction vector 1x3

        Parameters
        ----------
            image: image to run preprocessing on, should be the full resolution face cropped image
            landmarks: the facial landmarks
            pose: the head pose

        Returns
        -------
            left_eye, right_eye, head_direction_vector: the inputs needed by the model
        """
        left_eye = self.crop_eye(image, (landmarks['left_eye']['x']*image.shape[1], landmarks['left_eye']['y']*image.shape[0]))
        right_eye = self.crop_eye(image, (landmarks['right_eye']['x']*image.shape[1], landmarks['right_eye']['y']*image.shape[0]))
        head_direction_vector = [pose['yaw'], pose['pitch'], pose['roll']]
        # keys are specified by the models input layers
        inputs = {'head_pose_angles': head_direction_vector,
                  'left_eye_image': left_eye,
                  'right_eye_image': right_eye}
        return inputs

    def preprocess_output(self, raw_detections: np.ndarray) -> List[Dict[str, Any]]:
        """
        Change the format of the detections for use by the next model.

        Default model produces an output gaze vector 1x3

        Parameters
        ----------
            raw_detections: the gaze estimation network output as produced by OpenVINO

        Returns
        -------
            detections: a list of detections meeting the criteria
        """
        logging.info(f'raw detections {raw_detections}')
        gaze_vec = {'x': raw_detections['gaze_vector'][0][0].item(),
                    'y': raw_detections['gaze_vector'][0][1].item(),
                    'z': raw_detections['gaze_vector'][0][2].item()}
        return gaze_vec

    def visualize_gaze(self, image: np.ndarray, gaze_vec: List[Dict[str, float]], landmarks: Dict[str, Dict[str, float]]) -> np.ndarray:
        img = image.copy()
        # add the x vector and subtract the y vector, image coordinates x increases to the right
        # and y increases going down, guessing the gaze is using standard math coordinate system
        # where y increases going up
        left_eye = (int(landmarks['left_eye']['x']*img.shape[1]),
                    int(landmarks['left_eye']['y']*img.shape[0]))
        left_eye_gaze = (int(landmarks['left_eye']['x']*img.shape[1] + 100 * gaze_vec['x']),
                         int(landmarks['left_eye']['y']*img.shape[0] - 100 * gaze_vec['y']))

        right_eye = (int(landmarks['right_eye']['x']*img.shape[1]),
                     int(landmarks['right_eye']['y']*img.shape[0]))
        right_eye_gaze = (int(landmarks['right_eye']['x']*img.shape[1] + 100 * gaze_vec['x']),
                          int(landmarks['right_eye']['y']*img.shape[0] - 100 * gaze_vec['y']))
        img = cv2.line(img, left_eye, left_eye_gaze, (255, 0, 0), 5)
        img = cv2.line(img, right_eye, right_eye_gaze, (0, 0, 255), 5)

        return img

if __name__ == '__main__':
    # parse input arguments
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Estimate gaze')
    parser.add_argument('--input',
                        default='data/image_100_face.png',
                        type=str,
                        help='path to an image')
    parser.add_argument('--input-landmarks',
                        default='data/image_100_face_landmarks.json',
                        type=str,
                        help='path to the facial landmark json file')
    parser.add_argument('--input-pose',
                        default='data/image_100_head_pose.json',
                        type=str,
                        help='path to the head pose json file')
    parser.add_argument('--gaze',
                        default='models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002',
                        type=str,
                        help='path to the gaze estimation model')
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
    with open(args.input_landmarks, 'r') as f:
        landmarks = json.load(f)
    with open(args.input_pose, 'r') as f:
        pose = json.load(f)
    gaze_est = GazeEstimator(args.gaze, args.device)
    inputs = gaze_est.preprocess_input(image, landmarks, pose)
    dets = gaze_est.sync_detect(inputs)
    gaze_vec = gaze_est.preprocess_output(dets)
    img = gaze_est.visualize_gaze(image, gaze_vec, landmarks)
    cv2.imwrite(f'{args.output}/gaze_est.png', img)
    with open(f'{args.output}/gaze_vec.json', 'w') as f:
        json.dump(gaze_vec, f)

