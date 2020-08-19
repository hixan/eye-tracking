import cv2
import numpy as np
from project_tools import flush_buffer, get_angles, grab_frames
from pynput import mouse, keyboard  # input
from application import Application, ApplicationState
SAMPLE_SIZE = 1

class NoseControlBaseState(ApplicationState):

    def __init__(self, name: str, sample_size: int, scale_factor: int, cap: cv2.VideoCapture):
        """__init__.

        :param name: name of the state
        :param sample_size: number of images to sample and average
        :param scale_factor: scale down factor of image
        :param cap: cv2 videocapture device to retrieve images from
        :type cap: cv2.VideoCapture
        """

        super(NoseControlBaseState, self).__init__(name)

        # capture device
        self.cap = cap

        self.sample_size = sample_size
        self.scale_factor = scale_factor

    def loop(self):

        # grab frames
        ims, features = grab_frames(self.cap, self.sample_size, self.scale_factor)

        # hopefully denoising and not blurring
        img = np.mean(ims, axis=0).astype(np.uint8)

        # handle features
        if features is None:
            print('features was none')
            cv2.imshow('face', img[:, ::-1])
            k = cv2.waitKey(30) & 0xf
            if k == 27:
                return False  # exit program
            return True  # dont exit program but stop here for this frame

        # extract landmarks
        # center of left eye
        self.left_cent = np.mean(features['left_bb'], axis=0)
        # center of right eye
        self.right_cent = np.mean(features['right_bb'], axis=0)
        # between nostrils
        self.nose_cent = np.mean(features['nose'], axis=0)

        # angles of features triangle
        self.angles = [
            np.degrees(x) for x in get_angles(self.left_cent, self.right_cent, self.nose_cent)
        ]

        # show features and features triangle
        cv2.polylines(
            img,
            (features['left_bb'].astype(np.int32), features['right_bb'].astype(
                np.int32), features['nose'].astype(
                    np.int32), np.int32((self.left_cent, self.right_cent, self.nose_cent))),
            True, (255, 0, 0),
            thickness=1)
        # draw face
        cv2.imshow( 'face', img[:, ::-1])  # mirror to make it easier to see what is happening

        # cv2 needs this for some reason
        k = cv2.waitKey(30) & 0xf
        if k == 27:
            return False
        return True


class NoseControlCalibratorState(NoseControlBaseState):
    def __init__(self, sample_size, scale_factor, cap):
        super(NoseControlCalibratorState, self).__init__('calibrate', sample_size, scale_factor, cap)
        self.calibration = {}

        # TODO calculate these
        width = 2000
        height = 1900

        self.substates = [('left', (0, height//2)), ('right', (width, height//2))]
        self.substate = None

        self.bind(keyboard.Key.space, self.action)


    def action(self):
        if self.substate is not None:
            self.calibration[self.substate] = self.angles
            print(f'calibrated {self.substate}')

        if len(self.substates) == 0:
            print(f'Starting demo...')
            self.parent_app.change_state('demo')
            self.parent_app.calibration = self.calibration
            return
        # setup the next state
        self.substate, (x, y) = self.substates.pop()

        self.parent_app.indicate_location(x, y)
        

class NoseControlDemo(NoseControlBaseState):
    def __init__(self, sample_size: int, scale_factor: int, cap: cv2.VideoCapture):
        super(NoseControlDemo, self).__init__('demo', sample_size, scale_factor, cap)
        self.bind(keyboard.Key.space, self.indicate)
        self.bind(keyboard.KeyCode(char='0''0'), self.set_0)
        self.bind(keyboard.KeyCode(char='1'), self.set_1)
        self.bind(keyboard.KeyCode(char='2'), self.set_2)
        self.bind(keyboard.KeyCode(char='3'), self.set_average)
        self.index = [0]

    def set_0(self):
        print(' def set_0(self):')
        self.index = [0]

    def set_1(self):
        print(' def set_1(self):')
        self.index = [1]

    def set_2(self):
        print(' def set_2(self):')
        self.index = [2]

    def set_average(self):
        print(' def set_average(self):')
        self.index = [0, 1, 2]



    def indicate(self):
        cal = self.parent_app.calibration
        min = np.array(cal['left'])
        max = np.array(cal['right'])
        observed = np.array(self.angles)
        frac = (observed - min) / (max - min)
        x = np.mean(frac[self.index] * 2000)
        y = 1000
        self.parent_app.mouse_movement_duration = 0
        self.parent_app.indicate_location(x, y)

    def loop(self):
        super(NoseControlDemo, self).loop()
        self.indicate()


if __name__ == '__main__':

    app = Application({}, 'calibrate')
    base_state = NoseControlCalibratorState(3, 2, cv2.VideoCapture(0))
    demo_state = NoseControlDemo(3, 3, base_state.cap)
    app.add_state(base_state)
    app.add_state(demo_state)
    app.main()

