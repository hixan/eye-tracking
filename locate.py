import cv2
import numpy as np
from scipy.spatial import distance_matrix
from time import sleep
from munkres import Munkres
from itertools import permutations
import face_recognition
from matplotlib import pyplot as plt

SCALE_FACTOR = 3
cap = cv2.VideoCapture(2)


def create_bounding_box(contour, scale=1, min_area=False):
    if not min_area:
        x1 = np.min(contour[:, 0])
        x2 = np.max(contour[:, 0])
        y1 = np.min(contour[:, 1])
        y2 = np.max(contour[:, 1])

        return np.array([
            (x1, y2),
            (x1, y1),
            (x2, y1),
            (x2, y2),
        ])

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


def xy_polar(xy):
    rv = np.array((np.linalg.norm(xy, axis=1), np.arctan2(*xy.T))).T
    return rv


def polar_xy(polar):
    return (polar[:, 0] * np.array(
        (np.sin(polar[:, 1]), np.cos(polar[:, 1])))).T


def aligned_bounding_box(axis_alignment, contours, forced_aspect_ratio=.5):
    rv = []
    # calculate sin(angle) and cos(angle)
    denom = np.linalg.norm(axis_alignment)
    c, s = axis_alignment / denom

    # rotation matrix
    rot = np.array([[c, -s], [s, c]])
    counterrot = np.array([[c, s], [-s, c]])
    # transform points
    for contour in contours:
        rotated = contour @ rot
        box = create_bounding_box(rotated, min_area=False)
        if forced_aspect_ratio is not None:
            assert forced_aspect_ratio != 0, "aspect ratio cannot be 0"
            width = box[1][0] - box[2][0]
            height = box[3][1] - box[2][1]
            d_height = (width * forced_aspect_ratio - height) / 2
            box[0][1] = box[0, 1] - d_height
            box[1][1] = box[1, 1] + d_height
            box[2][1] = box[2, 1] + d_height
            box[3][1] = box[3, 1] - d_height

        # de-transform final array
        rv.append(box @ counterrot)
    return rv


face = None
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while True:

    # flush buffer, (setting buffer size can fail)
    for _ in range(int(cap.get(cv2.CAP_PROP_BUFFERSIZE))):
        cap.grab()

    # read and process image
    ret, img = cap.read()
    small = img[::SCALE_FACTOR, ::SCALE_FACTOR, :]
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    features = face_recognition.face_landmarks(small)

    # handle features
    if len(features) == 1:

        # extract / compute wanted features
        face_instance = {
            k: np.array(features[0][k]).astype(np.float64) * SCALE_FACTOR
            for k in ('chin', 'left_eye', 'right_eye', 'nose_tip')
        }
        horizontal = face_instance['chin'][-1] - face_instance['chin'][0]
        for eyekey in 'left_eye', 'right_eye':
            face_instance[eyekey + '_bb'] = aligned_bounding_box(
                horizontal, [face_instance[eyekey]])[0].astype(np.float64)
        face = face_instance

    # draw and extract facial features
    if face is not None:
        for fkey in face:
            feature = np.int32(face[fkey])

            if 'bb' in fkey:
                is_eye = True
                p1, p2, p3, p4 = feature
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

            if 'bb' in fkey:
                cv2.polylines(img, [feature],
                              is_eye, (255, 255, 255),
                              thickness=1)

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
    sleep(1)
cap.release()
cv2.destroyAllWindows()
