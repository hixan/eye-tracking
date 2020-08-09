import pyautogui  # screengrab
import sys
from pynput import mouse, keyboard  # input
from time import sleep
import numpy as np
from datetime import datetime
import cv2
from pathlib import Path
import json
import project_tools

# initialize constants
SIZE = np.array(pyautogui.size())
SAMPLE_SIZE = 5
DATADIR = Path('raw_data') / datetime.now().strftime('%y%m%d-%H:%M:%S')

# raise error if overwriting folder
DATADIR.mkdir(parents=True, exist_ok=False)
with open(DATADIR / 'metadata.json', 'w') as f:
    json.dump({
        'SAMPLE_SIZE': SAMPLE_SIZE,
        'SCREEN_SIZE': SIZE.tolist(),
    }, f)

# initialize globals
RUN = True
IMAGE_COUNTER = 0
SEARCHING = True
CAPS = [cv2.VideoCapture(0)]


def take_snapshot(cap):
    global IMAGE_COUNTER
    project_tools.flush_buffer(cap)
    im_names = []
    ims = []

    # capture images
    for ret, im in (cap.read() for i in range(SAMPLE_SIZE)):
        if not ret:
            raise BufferError('Could not read frame')
        name = f'{IMAGE_COUNTER:0>4}.jpg'
        im_names.append(name)
        ims.append(im)
        IMAGE_COUNTER += 1

    # show images
    for name, im in zip(im_names, ims):
        cv2.imwrite(str(DATADIR / name), im)
        cv2.imshow('capture', np.concatenate(ims, axis=0)[::3, ::3])
        cv2.waitKey(1)

    return im_names


def on_press(key):
    if key == keyboard.Key.esc:
        global RUN
        RUN = False
        print('Escape pressed. Ending session.')
    if key == keyboard.Key.space:
        global SEARCHING
        SEARCHING = False
        print('Space pressed. Taking snapshot.')


def gen_random_locations(size):
    ''' generate random locations that are not over dark parts of the screen '''
    im = np.mean(pyautogui.screenshot(), axis=2)
    sample = np.uint0(np.random.rand(size * 2, 2) * SIZE)
    try:
        return sample[im[tuple(sample.T[::-1])] > 3][:size]
    except IndexError:
        # if there are a lot of black pixels try with larger sample
        return gen_random_locations(size * 2)[:size]


def get_column(greyscale,
               column_index,
               horisontal_offset=0,
               vertical_offset=0):
    print()
    print('greyscale shape', greyscale.shape)
    print('column_index', column_index)
    xloc = column_index + horisontal_offset
    print(np.argmax(greyscale[:, column_index] > 0))
    top = xloc, np.argmax(greyscale[:, column_index] > 0) + vertical_offset
    # max index - first true in reversed
    bottom = xloc, greyscale.shape[0] - 1 - np.argmax(
        greyscale[:, column_index::-1] > 0) - vertical_offset
    middle = xloc, (top[1] + bottom[1]) / 2
    print(top, middle, bottom)
    print()
    return top, middle, bottom


def get_calibration_data(boarder, duration=3):
    screenshot_grey = np.sum(pyautogui.screenshot(), axis=2)
    print(screenshot_grey.shape)
    # first column with non-null pixels
    left_col = np.argmax(np.sum(screenshot_grey, axis=0) > 0)
    right_col = screenshot_grey.shape[1] - 1 - np.argmax(
        np.sum(screenshot_grey[::-1], axis=0) > 0)
    middle_col = (left_col + right_col) // 2
    tl, ml, bl = get_column(screenshot_grey, left_col, boarder, boarder)
    tm, mm, bm = get_column(screenshot_grey, middle_col, boarder, boarder)
    tr, mr, br = get_column(screenshot_grey, right_col, boarder, boarder)
    print('left', tl, ml, bl, '\nmiddle', tm, mm, bm, '\nright', tr, mr, br)

    locations = [tr, mr, br, tm, mm, bm, tl, ml, bl]
    for location in locations:
        pyautogui.moveTo(screenshot_grey.shape[1] // 2,
                         screenshot_grey.shape[0] // 2,
                         duration=0)
        pyautogui.moveTo(*location, duration=.5)
    print(locations)


get_calibration_data(0)
assert 0

locs = gen_random_locations(500)  # maximum number of obs
keyboard.Listener(on_press=on_press).start()
obs = []

for loc in locs:
    pyautogui.moveTo(*loc, duration=0.5)
    SEARCHING = True
    while RUN and SEARCHING:
        sleep(0.1)
        if not SEARCHING:
            for cap in CAPS:
                images = take_snapshot(cap)
                actual_loc = pyautogui.position()
                print('took', images, 'at', actual_loc)
                obs.append({'location': actual_loc, 'images': images})

    if not RUN:
        break

with open(DATADIR / 'observations.json', 'w') as f:
    json.dump(obs, f)
