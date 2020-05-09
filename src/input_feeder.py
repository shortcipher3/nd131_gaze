'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder('video.mp4')
    for batch in feed:
        do_something(batch)
    feed.close()
'''

import cv2

class InputFeeder:
    def __init__(self, video_stream):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        video_stream: video file or image file sequence or a capturing device or an IP video stream
            for video capturing,
            Can be a path to an mp4 (str), a url (str), a path to an image (str), or a v4l2 device
            such as a webcam (int - may be represented as string)
        '''
        self.video_stream = video_stream
        try:
            int(video_stream)
            vc = cv2.VideoCapture(int(video_stream))
        except:
            vc = cv2.VideoCapture(video_stream)
        if not vc.isOpened():
            raise IOError("Unable to read the specified video stream " + video_stream)
        self.vc = vc
        self.frame_count = 0
        self.frame = None

    def get_shape(self):
        return int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_fps(self):
        return self.vc.get(cv2.CAP_PROP_FPS)

    def __next__(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        got_flag, frame = self.vc.read()
        if got_flag:
            self.frame = frame
            self.frame_count += 1
        elif self.frame_count != 1:
            raise StopIteration()
        return self.frame

    def __iter__(self):
        return self

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        self.vc.release()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract n frames from a video source')
    parser.add_argument('--input',
                        default='bin/demo.mp4',
                        type=str,
                        help='open video file or image file sequence or a capturing device or an IP video stream for video capturing')
    parser.add_argument('--frames',
                        default=10,
                        type=int,
                        help='number of frames to extract')

    args = parser.parse_args()

    inp = InputFeeder(args.input)

    for k in range(args.frames):
        frame = next(inp)
        cv2.imwrite(f'image_{k:03}.png', frame)



