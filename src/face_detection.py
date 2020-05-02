'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import time
import cv2
from openvino.inference_engine import IECore

class FaceDetector:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model = model_name
        self.device = device
        model_weights = self.model + '.bin'
        model_structure = self.model + '.xml'

        ### Load model ###
        #t1 = cv2.getTickCount()
        core = IECore()
        self.net = core.read_network(model=model_structure, weights=model_weights)
        self.exec_net = core.load_network(network=self.net, device_name=self.device)
        #t2 = cv2.getTickCount()
        #print(f'Time taken to load model = {(t2-t1)/cv2.getTickFrequency()} seconds')

        ### Check model ###
        # Get the supported layers of the network
        supported_layers = core.query_network(network=self.net, device_name=self.device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        # Get the input layer
        self.input_blob = next(iter(self.exec_net.inputs))
        self.output_blob = next(iter(self.exec_net.outputs))

    def async_detect(self, batch):
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
            detections: the network's detections
        """
        while True:
            status = infer_request_handle.wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        detections = infer_request_handle.outputs
        return detections

    def sync_detect(self, image):
        """
        Synchronously run prediction on a batch with the network

        Parameters
        ----------
            batch: the batch of images to perform inference on

        Returns
        -------
            detections_arr: the array of detections
        """
        #t1 = cv2.getTickCount()
        detections = self.exec_net.infer({self.input_blob: image})
        #t2 = cv2.getTickCount()
        #print(f'Time taken to execute model = {(t2-t1)/cv2.getTickFrequency()} seconds')
        return detections

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError

    def draw_bboxes(image, detections):
        img = image.copy()
        for i in range(detections['batch'].shape[0]):
            #classId = int(detections['class'][i])
            score = float(detections['score'][i])
            bbox = [float(v) for v in detections['bbox'][i]]
            if score > 0.3:
                #print(f"batch: {detections['batch'][i]} class: {classId}, score: {score}, bbox: {bbox}")
                y = bbox[1] * img.shape[0]
                x = bbox[0] * img.shape[1]
                bottom = bbox[3] * img.shape[0]
                right = bbox[2] * img.shape[1]
                cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
        return img

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Use computer vision to control your mouse position with your gaze')
    parser.add_argument('--input',
                        default='data/image_100.png',
                        type=str,
                        help='open video file or image file sequence or a capturing device or an IP video stream for video capturing')
    parser.add_argument('--detection',
                        default='models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
                        type=str,
                        help='path to the face detection model')
    parser.add_argument('--device',
                        default='CPU',
                        type=str,
                        help='the device to run inference on, one of CPU, GPU, MYRIAD, FPGA')

    args = parser.parse_args()

    face_det = FaceDetector(args.detection, args.device)

