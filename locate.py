import cv2
import numpy as np
from scipy.spatial import distance_matrix
from time import sleep
from munkres import Munkres
from itertools import permutations
from matplotlib import pyplot as plt
from project_tools import flush_buffer, aligned_bounding_box, get_features

global SCALE_FACTOR
SCALE_FACTOR = 3
SAMPLE_SIZE = 1


def extract_box(im, box):
    w = np.linalg.norm(box[1] - box[2])
    h = np.linalg.norm(box[2] - box[3])

    dst = np.float32([(0, 0), (0, h - 1), (w - 1, h - 1), (w - 1, 0)])

    m = cv2.getPerspectiveTransform(np.float32([box[np.array([2, 3, 0, 1])]]),
                                    dst)
    return cv2.warpPerspective(
        im,  #
        m,  #
        (int(w), int(h))  #
    )


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    face = None
    flush = not cap.set(cv2.CAP_PROP_BUFFERSIZE, SAMPLE_SIZE)
    flush = False

    while True:
        if flush:
            flush_buffer(cap)
        ims = [cap.read()[1] for _ in range(SAMPLE_SIZE)]
        img = ims[SAMPLE_SIZE // 2]
        try:
            features = get_features(ims)
        except ValueError:
            cv2.imshow('face', img)
            continue
        left = extract_box(img, features['left_bb'])
        right = extract_box(img, features['right_bb'])

        cv2.polylines(img,
                      np.int32([features['left_bb'], features['right_bb']]),
                      True, (255, 255, 255),
                      thickness=1)
        cv2.imshow(
            'face',
            img[:, ::-1])  # mirror to make it easier to see what is happening
        cv2.imshow('left', left)
        cv2.imshow('right', right)
        k = cv2.waitKey(30) & 0xf
        if k == 27:
            break

    # while False:
    #
    #     # flush buffer, (setting buffer size can fail)
    #     flush_buffer(cap)
    #
    #     # read and process image
    #     ret, img = cap.read()
    #     small = img[::SCALE_FACTOR, ::SCALE_FACTOR, :]
    #     features = face_recognition.face_landmarks(small)
    #
    #     # handle features
    #     if len(features) == 1:
    #
    #         # extract / compute wanted features
    #         face_instance = {
    #             k: np.array(features[0][k]).astype(np.float64) * SCALE_FACTOR
    #             for k in ('chin', 'left_eye', 'right_eye', 'nose_tip')
    #         }
    #         horizontal = face_instance['chin'][-1] - face_instance['chin'][0]
    #         for eyekey in 'left_eye', 'right_eye':
    #             face_instance[eyekey + '_bb'] = aligned_bounding_box(
    #                 horizontal, [face_instance[eyekey]])[0].astype(np.float64)
    #         face = face_instance
    #
    #     # draw and extract facial features
    #     if face is not None:
    #         for fkey in face:
    #             feature = np.int32(face[fkey])
    #
    #             if 'bb' in fkey:
    #                 is_eye = True
    #                 p1, p2, p3, p4 = feature
    #                 w = np.linalg.norm(p4 - p1)
    #                 h = np.linalg.norm(p3 - p4)
    #                 hw = np.int32((w, h))
    #
    #                 dst = np.float32([(0, 0), (0, h - 1), (w - 1, h - 1),
    #                                   (w - 1, 0)])
    #                 m = cv2.getPerspectiveTransform(np.float32([p3, p4, p1, p2]),
    #                                                 dst)
    #
    #                 ret = cv2.warpPerspective(img, m, tuple(hw))
    #                 cv2.imshow(fkey, ret)
    #             else:
    #                 is_eye = 'eye' in fkey
    #
    #             if 'bb' in fkey:
    #                 cv2.polylines(img, [feature],
    #                               is_eye, (255, 255, 255),
    #                               thickness=1)
    #
    #     # draw bounding box on face
    #     # gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    #     # locations = face_recognition.face_locations(gray_small)
    #     # for y1, x1, y2, x2 in locations:
    #     #     tr = np.array([x1, y1]) * SCALE_FACTOR
    #     #     bl = np.array([x2, y2]) * SCALE_FACTOR
    #     #     cv2.rectangle(img, tuple(tr), tuple(bl), (0, 0, 255))
    #     cv2.imshow('img', img[:, ::-1, :])
    #     k = cv2.waitKey(30) & 0xf
    #     if k == 27:
    #         break
    cap.release()
    cv2.destroyAllWindows()
