import numpy as np
import cv2
from application import Cv2CaptureApplicationState, Cv2CaptureApplication
from typing import Dict, Tuple, Optional
# from my_tools.tools import debug_print as print
from my_tools.pipeline import FastFacialFeatures

# Much of this file was deleloped with the help of the following blog post:
# https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

reference_points = np.array([
    (     0,      0,      0), # nose tip
    (   0.0, -330.0,  -65.0), # chin
    (-225.0,  170.0, -135.0), # left eye
    ( 225.0,  170.0, -135.0), # right eye
    (-150.0, -150.0, -125.0), # left mouth
    ( 150.0, -150.0, -125.0), # right mouth
])


def camera_matrix(cal_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # returns camera matrix and distortion coefficients. These are all simple estimates
    cam_mat = np.diag((*cal_image.shape[:2], 1))
    cam_mat[:2, 2] = np.array(cal_image.shape[:2]) // 2
    dist_coefficients = np.zeros((4, 1))
    return cam_mat.astype(np.int32), dist_coefficients.astype(np.int32)


# I checked the radial distortion of my webcam, there is some but very little

# dont really understand scaling factor


class HeadPoseState(Cv2CaptureApplicationState):
    def __init__(self, name: str):
        super(HeadPoseState, self).__init__(name)
        self._features_model = FastFacialFeatures(0.5, True)

    def application_init(self):
        self.camera_matrix, self.distance_coefficients = camera_matrix(
            self.getframes(1)[0])

    def loop(self):
        # first image from get frames
        imgs = self.getframes()

        # extract faces from image
        all_features = []
        for img in imgs:
            features = self._features_model.transform(img)
            if len(features) == 0:
                print('could not find any faces!')
                cv2.imshow('frame', imgs[1][:, ::-1])
                return
            else:
                all_features.append(features[0])

        # average over all features
        averaged_features = all_features[0]
        for feat in all_features[1:]:
            for fname in feat:
                averaged_features[fname] += feat[fname]
        for fname in averaged_features:
            averaged_features[fname] /= len(all_features)

        # predict the pose from the averaged features
        rot_vec, trans_vec = self.predict_pose(averaged_features)

        # draw an indication of direction
        nose_end_point2D, jacobian = cv2.projectPoints(
            np.array([(0, 0, 1000)], dtype='double'), rot_vec.astype('double'),
            trans_vec.astype('double'), self.camera_matrix.astype('double'),
            self.distance_coefficients.astype('double'))
        dir_points = np.array(
            [averaged_features['nose_bridge'][-1], nose_end_point2D[0][0]],
            dtype=np.int64)
        self.draw_location(imgs[1], dir_points)

        # draw facial features
        for feat in averaged_features.values():
            self.draw_location(imgs[1],
                               feat.astype(np.int32),
                               thickness=1,
                               is_closed=False)

        # show the frame
        cv2.imshow('frame', imgs[1][:, ::-1])

    def draw_location(self,
              img: np.ndarray,
        locations: np.ndarray,
        thickness: int = 3,
        is_closed: bool = True,
            color: Tuple[int, int, int] = (0, 255, 255)
    ):
        assert (locations[0].dtype == np.int64 or locations[0].dtype == np.int32,
                'draw_location needs locations[n].dtype to be int32 or int64')
        # draw a circle at the location if there is only one point
        if len(locations) == 1:
            cv2.circle(img,
                       tuple(locations[0]),
                       thickness,
                       color,
                       thickness=-1)
        else:
            # otherwise draw the entire shape
            cv2.polylines(img, [locations], is_closed, color, thickness)

    def predict_pose(
        self,
        features: Dict[str, np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """predict_pose.

        :param features: facial landmarks as returned by face_recognition
        :type features: Dict[str, np.ndarray]
        """
        image_points = np.array([
            features['nose_bridge'][-1],  # nose tip
            features['chin'][8],  # chin
            features['left_eye'][0],  # left eye
            features['right_eye'][3],  # right eye
            features['top_lip'][0],  # left mouth
            features['top_lip'][6],  # right mouth
        ])
        success, rot_vec, trans_vec = cv2.solvePnP(
            reference_points, image_points,
            self.camera_matrix, self.distance_coefficients
        )
        if success:
            return rot_vec, trans_vec


app = Cv2CaptureApplication('main_state', 10, 0)
app.add_state(HeadPoseState('main_state'))
app.main()
