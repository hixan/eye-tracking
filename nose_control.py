import cv2
import numpy as np
from project_tools import flush_buffer, get_features, get_angles, grab_frames
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
        width = 2500
        height = 1900

        self.substates = [('left', (0, height//2)), ('right', (width, height//2))]
        self.substate = None

        self.bind(keyboard.Key.space, self.action)


    def action(self):
        print('called')
        if self.substate is not None:
            self.calibration[self.substate] = self.angles

        if len(self.substates) == 0:
            self.parent_app.change_state('demo')
            return
        # setup the next state
        self.substate, (x, y) = self.substates.pop()

        self.parent_app.indicate_location(x, y)
        

class NoseControlDemo(NoseControlBaseState):
    def __init__(self, sample_size: int, scale_factor: int, cap: cv2.VideoCapture):
        super(NoseControlDemo, self).__init__('demo', sample_size, scale_factor, cap)
        self.bind(keyboard.Key.space, self.indicate)

    def indicate(self):
        print('TODO!')


def abc():
    '''
# class NoseControlCalibrator(NoseControlAppBase):
# 
#     def __init__(self, sample_size, scale_factor, screen_width, screen_height, cap):
#         super(NoseControlCalibrator, self).__init__(sample_size, scale_factor, cap=cap)
#         self.states = [
#                 ('left',   (0 * screen_width, 0.5 * screen_height)),
#                 ('right', (1 * screen_width, 0.5 * screen_height))
#         ]
#         self.overall_state = 'Calibrate'
#         self.state = None
#         self.calibration = {}
# 
#     def positive_action(self):
#         pass
# 
#     def positive_action_calibration(self):
#         if self.state is not None:
#             self.calibration[self.state] = self.angles
# 
#         if len(self.states) == 0:
#             self.overall_state = 'Main'
#             return
#         # setup the next state
#         self.state, (x, y) = self.states.pop()
# 
#         self.indicate_location(x, y)
# 
#     def positive_action_main(self):
#         pass
#     def main_calibration(self):
#         pass
#     def main_application(self):
#         pass
# 
# class NoseControlApplication(NoseControlAppBase):
# 
#     def __init__(self, sample_size, scale_factor):
#         super(NoseControlApplication, self).__init__(sample_size, scale_factor)
#         self.calibrator = NoseControlCalibrator(
#                 sample_size * 2,
#                 scale_factor // 2,
#                 2000, 1700,
#                 cap=self.cap)
# 
#     def positive_action(self):
#         left = self.calibrator.calibration['left'][2]
#         right = self.calibrator.calibration['right'][2]
#         current = self.angles[2]
# 
#         frac = (current - left) / (right - left)
# 
# 
# 
# 
#     def main(self):
#         self.calibrator.main()
#         super(NoseControlApplication, self).main()
# 
'''
    pass


if __name__ == '__main__':

    app = Application({}, 'calibrate')
    base_state = NoseControlCalibratorState(1, 2, cv2.VideoCapture(0))
    demo_state = NoseControlDemo(1, 4, base_state.cap)
    app.add_state(base_state)
    app.add_state(demo_state)
    app.main()

