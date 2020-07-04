import cv2
import numpy as np
from scipy.spatial import distance_matrix
from munkres import Munkres
from itertools import permutations
import face_recognition
from matplotlib import pyplot as plt

SCALE_FACTOR = 3
cap = cv2.VideoCapture(2)


def create_bounding_box(contour, scale=1):
    # returns a bounding box with the uppermost corner on the left, and the
    # width larger then the height.
    rect = cv2.minAreaRect(np.int0(contour))
    boundary = np.array(cv2.boxPoints(rect))

    # shift along so the first point is the highest
    shift = np.argmax(boundary[:, 1])

    # if width is smaller then height, shift along one extra
    if np.linalg.norm(boundary[shift] - boundary[shift - 1]) < np.linalg.norm(
            boundary[shift - 1] - boundary[shift - 2]):
        shift += 1

    rv = boundary[np.arange(shift - 4, shift, 1)]

    # increase scale
    if scale != 1:
        cent = np.mean(rv, axis=0)
        dists = rv - cent
        return dists * scale + cent
    return rv


def xy_rtheta(xy):
    rv = np.array((np.linalg.norm(xy, axis=1), np.arctan2(*xy.T))).T
    return rv


def rtheta_xy(rtheta):
    return (rtheta[:, 0] * np.array(
        (np.sin(rtheta[:, 1]), np.cos(rtheta[:, 1])))).T


'''
def apply_inertia(old_locs, new_locs, inertia=0.9, center_inertia=0.):
    # ignore centers
    new_cent = np.mean(new_locs, axis=0)
    old_cent = np.mean(old_locs, axis=0)
    #print('old_cent.shape', old_cent.shape)

    new_shape = new_locs - new_cent
    old_shape = old_locs - old_cent
    #print('old_shape.shape', old_shape.shape)

    new_shape_polar = xy_rtheta(new_shape)
    old_shape_polar = xy_rtheta(old_shape)
    #print('old_shape_polar.shape', old_shape_polar.shape)

    new_mid_rot = np.mean(new_shape_polar[:, 1])
    old_mid_rot = np.mean(old_shape_polar[:, 1])
    #print('old_mid_rot.shape', old_mid_rot.shape)

    new_shape_polar_rot = new_shape_polar - np.array([(0, new_mid_rot)])
    old_shape_polar_rot = old_shape_polar - np.array((0, old_mid_rot))
    #print('old_shape_polar_rot.shape', old_shape_polar_rot.shape)

    # inertia
    ret_shape_polar_rot = inertia * old_shape_polar_rot + (
        1 - inertia) * new_shape_polar_rot
    #print('ret_shape_polar_rot.shape', ret_shape_polar_rot.shape)

    ret_shape_polar = ret_shape_polar_rot
    ret_shape_polar[:, 1] += new_mid_rot
    #print('ret_shape_polar.shape', ret_shape_polar.shape)

    ret_shape = rtheta_xy(ret_shape_polar)
    #print('ret_shape.shape', ret_shape.shape)
    ret = ret_shape + new_cent
    #print('ret.shape', ret.shape)
    #print()
    return ret

    #return old_locs * inertia + (1 - inertia) * new_locs
    return (old_shape * inertia + (1 - inertia) * new_shape
            ) + old_cent * center_inertia + (1 - center_inertia) * new_cent

'''

face = None

while True:

    # read and process image
    ret, img = cap.read()
    small = img[::SCALE_FACTOR, ::SCALE_FACTOR, :]
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    features = face_recognition.face_landmarks(small)

    # handle features
    if len(features) == 1:

        # extract / compute wanted features
        face_instance = {
            k: np.array(features[0][k]).astype(np.float64)
            for k in ('chin', 'left_eye', 'right_eye', 'nose_tip')
        }
        for eyekey in 'left_eye', 'right_eye':
            face_instance[eyekey + '_bb'] = create_bounding_box(
                face_instance[eyekey]).astype(np.float64)

        if face is None:  # no inertia to add
            face = face_instance
        else:
            for k in face:
                face[k] = apply_inertia(face[k], face_instance[k])

    # draw and extract facial features
    if face is not None:
        for fkey in face:
            feature = np.int32(face[fkey])

            if 'bb' in fkey:
                is_eye = True
                p1, p2, p3, p4 = feature * SCALE_FACTOR
                cv2.circle(img, tuple(p1.astype(np.int0)), 2, (255, 0, 0), -1)
                w = np.linalg.norm(p4 - p1)
                h = np.linalg.norm(p3 - p4)
                hw = np.int32((w, h))

                dst = np.float32([(0, 0), (0, h - 1), (w - 1, h - 1),
                                  (w - 1, 0)])
                m = cv2.getPerspectiveTransform(np.float32([p3, p4, p1, p2]),
                                                dst)

                ret = cv2.warpPerspective(img, m, tuple(hw))
                cv2.imshow(fkey, ret)
            else:
                is_eye = 'eye' in fkey

            cv2.polylines(img, [feature * SCALE_FACTOR],
                          is_eye, (255, 255, 0 if 'bb' not in fkey else 255),
                          thickness=5)

    # draw bounding box on face
    locations = face_recognition.face_locations(gray_small)
    for y1, x1, y2, x2 in locations:
        tr = np.array([x1, y1]) * SCALE_FACTOR
        bl = np.array([x2, y2]) * SCALE_FACTOR
        cv2.rectangle(img, tuple(tr), tuple(bl), (0, 0, 255))
    cv2.imshow('img', img[:, ::-1, :])
    k = cv2.waitKey(30) & 0xf
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
