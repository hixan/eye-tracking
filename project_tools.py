import cv2
import face_recognition
from operator import itemgetter
import numpy as np

first_last = itemgetter(0, -1)


def flush_buffer(cap, keep=1):
    """remove old frames from a capture devices buffer.

    :param cap: capture device
    :param keep: number of frames to leave in the capture devices buffer
    """
    for _ in range(int(cap.get(cv2.CAP_PROP_BUFFERSIZE)) - keep):
        cap.grab()


def create_bounding_box(contour, scale: float = 1., allow_rotation: float = False):
    """Create a bounding box around a contour.

    :param contour: ordered list of x,y pairs defining contour outline
    :param scale: scale the size of the output by this amount (about center by mean)
    :type scale: float
    :param allow_rotation: True allows rotation to make the area smaller
    :type allow_rotation: float
    """

    if not allow_rotation:
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


def aligned_bounding_box(axis_alignment, contours, forced_aspect_ratio=0.5):
    """create a bounding box aligned with the vector <axis_alignment> that contains the whole contour.

    :param axis_alignment: axis to align all bounding boxes by
    :param contours: list of contours to create bounding boxes for
    :param forced_aspect_ratio: aspect ratio of the output box (height / width)
    """
    # sanitise inputs
    assert forced_aspect_ratio != 0, "aspect ratio cannot be 0"

    rv = []

    # calculate sin(angle) and cos(angle) by pythagoras
    c, s = axis_alignment / np.linalg.norm(axis_alignment)

    # rotation matrix to rotate clockwise by the angle
    rot = np.array([[c, -s], [s, c]])
    # rotation matrix to rotate counterclockwise by the angle
    counterrot = np.array([[c, s], [-s, c]])
    # transform points to make bounding box oriented with desired axes
    for contour in contours:
        rotated = contour @ rot
        box = create_bounding_box(rotated)
        if forced_aspect_ratio is not None:
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


def get_features(imagelist, scale_factor: int = 1):
    ''' extract desired features from the image with face_recognition module.

    :param imagelist: list of images to extract faces from
    :param scale_factor: factor by which to scale (must be int currently TODO scale properly?)'''

    raw_faces = []
    # extract features
    for img in imagelist:
        small = img[::scale_factor, ::scale_factor, :]
        features = face_recognition.face_landmarks(small)

        if len(features) == 0:
            raise ValueError('No Faces Found in one of the images')
        assert len(features) == 1, 'multiple faces found.'

        face_raw = {
            k: np.float32(features[0][k])
            for k in ('left_eye', 'right_eye', 'nose_tip')
        }
        f, l = np.array(first_last(features[0]['chin']))
        face_raw['h_vector'] = l - f
        raw_faces.append(face_raw)

    features = {}
    # calculate overall horizontal vector
    hvec = np.average(np.array(list(map(itemgetter('h_vector'), raw_faces))),
                      axis=0)
    features['hvec'] = hvec

    # calculate eye boxes
    bbs_left = np.array(
        aligned_bounding_box(hvec, map(itemgetter('left_eye'), raw_faces),
                             0.2)) * scale_factor
    bbs_right = np.array(
        aligned_bounding_box(hvec, map(itemgetter('right_eye'), raw_faces),
                             0.5)) * scale_factor

    # average the sizes of the eye boxes
    features['left_bb'] = np.average(bbs_left, axis=0)
    features['right_bb'] = np.average(bbs_right, axis=0)

    # average nose position
    #features['nose'] = np.mean(np.array(
    #    list(map(itemgetter('nose_tip'), raw_faces))),
    #                           axis=0)
    features['nose'] = (raw_faces[-1]['nose_tip'] * scale_factor).astype(
        np.float64)  # TODO make average / median

    # take the eyes from the median image
    im = imagelist[len(imagelist) // 2]

    return features


def get_angles(A, B, C):
    ''' uses cosine rule to calculate angles at each vertex defined by the two neighbours '''
    # edge length opposite to corner with the same label
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(B - A)
    print(2 * b * c * (a**2 - b**2 - c**2))

    angleC = np.arccos((a**2 + b**2 - c**2) / 2 / a / b)
    angleA = np.arccos((c**2 + b**2 - a**2) / 2 / c / b)
    angleB = np.arccos((a**2 + c**2 - b**2) / 2 / a / c)

    return angleA, angleB, angleC
