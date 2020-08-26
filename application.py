from pynput import keyboard
from typing import Callable
from collections import defaultdict
import traceback
import cv2
import pyautogui
import abc


class ApplicationState:
    '''Main application instance.
    Keeps track of keypresses and the active program state.'''

    def __init__(self, name: str):
        """ApplicationState

        :param name: name of this application state
        :type name: str
        """
        # binds keys to actions
        self.binder = defaultdict(lambda: [])
        self.parent_app: Application = None
        self.name = name

    def application_init(self):
        pass

    @property
    def parent_app(self):
        if self._parent_app is not None:
            return self._parent_app
        raise RuntimeError(f'{self} tried to access parent_app before it was set!')

    @parent_app.setter
    def parent_app(self, value):
        self._parent_app = value

    @abc.abstractmethod
    def loop(self):
        '''Main loop of this program, must be implemented.'''
        raise NotImplementedError

    def change_state(self, state):
        """change_state from this state to the state with name <state>.

        :param state: the name of the state to change to
        """
        if self.parent_app is None:
            raise ValueError(f'State {self.name} tried to change state when there is no parent')
        return self.parent_app.change_state(state)

    def __str__(self):
        return f'ApplicationState({self.name})'

    def bind(self, key, function: Callable):
        """bind.

        :param key: keycode (as described by pynput.keyboard.Key)
        :param function: function to call on key press
        :type function: Callable
        """
        self.binder[key].append(function)


class Application:
    """main application instance. Handles keypresses."""

    def __init__(self, start_state: str):

        # global keybindings (default dict handles the case where there is no binding)
        self.binder = defaultdict(lambda: [], {keyboard.Key.esc: [self.exit]})

        # application states
        self.states = {}
        self.state = start_state

        # default values
        self.mouse_movement_duration = 0.2

        # flag to exit application
        self._exit = False

    @property
    def active_keys(self):
        """keys with a keybinding"""
        return self.binder.keys()

    def _onpress(self, key):
        """handles keyboard input; to pass to pynput."""
        # if key in self.active_keys or True:
        #     print(f'{key} pressed')
        for action in self.binder[key] + self.states[self.state].binder[key]:
            try:
                action()
            except:
                traceback.print_exc()
                print('The above error occured. Continuing...')

    def exit(self):
        """Exit the application."""
        print('Exiting...')
        self._exit = True

    def _frame(self):
        """Calls the correct states main loop method followed by the optional loop method."""
        try:
            self.states[self.state].loop()
        except:
            print(f'failed to execute state {self.state} loop.')
            raise

        try:
            self.loop()
        except:
            print('failed to execute application loop.')
            raise

    def indicate_location(self, x, y):
        """indicate_location.

        :param x: x coordinate
        :param y: y coordinate
        """

        pyautogui.moveTo(x, y, self.mouse_movement_duration)

    def get_location(self, x, y):
        return pyautogui.location()

    def main(self):
        """main."""
        # start keyboard listener
        listener = keyboard.Listener(on_press=self._onpress)
        listener.start()
        if self.state not in self.states:
            raise ValueError(f'Starting state {self.state} not defined!')
        # main loop
        while True:
            self._frame()
            if self._exit:
                break
        listener.stop()

    def change_state(self, state):
        assert state in self.states, f'tried to change to a non-existant state ({state}) options are {self.states.keys()}'
        self.state = state

    def add_state(self, state: ApplicationState):
        assert state.name not in self.states, f'state {state} already in application!'

        self.states[state.name] = state
        state.parent_app = self
        state.application_init()


class Cv2CaptureApplication(Application):
    def __init__(self, starting_state: str, default_framecount: int, *capture_args, **capture_kwargs):
        """__init__.

        :param starting_state: name of state to enter when program begins
        :type starting_state: str
        :param default_framecount: sets the default number of frames to be returned by getframes
        :type default_framecount: int
        :param capture_args: args to be passed to cv2.VideoCapture
        :param capture_kwargs: kwargs to be passed to cv2.VideoCapture
        """
        self._cap = cv2.VideoCapture(*capture_args, **capture_kwargs)
        assert self._cap.isOpened(), f'could not open capture device with {capture_args} and {capture_kwargs}'
        self._set_cap_buffer(default_framecount)

        super(Cv2CaptureApplication, self).__init__(starting_state)

    def _set_cap_buffer(self, buffersize):
        # attempt to set buffer size and save success state for later
        self._cap_buffersize = int(self._cap.get(cv2.CAP_PROP_BUFFERSIZE))
        print('self._cap_buffersize', self._cap_buffersize)
        # flag to flush buffer
        self._buffer_difference = int(self._cap_buffersize - buffersize)
        print('self._buffer_difference', self._buffer_difference)
        assert (self._buffer_difference >= 0,
                'cap buffer size could not be adjusted, and cannot be flushed to the correct size!')

    def loop(self):
        cv2.waitKey(1)

    def getframes(self, framecount: int = None, reset_default: bool = True):
        """retrieves frames. If framecount is None, retrieves the default amount.

        :param framecount: number of frames to retrieve (falsey retrieves the default amount). If greater then default, the default is returned.
        :type framecount: int
        :param reset_default: if framecount is not None, sets default amount to framecount
        :type reset_default: bool
        """
        if framecount and reset_default:
            if self._cap_buffersize != framecount:
                self._set_cap_buffer(framecount)
        if framecount:
            flushnumber = self._cap_buffersize - framecount
        else:
            flushnumber = self._buffer_difference

        for _ in range(flushnumber):
            self._cap.grab()

        ims = [self._cap.read()[1] for _ in range(self._cap_buffersize - flushnumber)]

        return ims


class Cv2CaptureApplicationState(ApplicationState):

    def getframes(self, framecount: int = None, reset_default: bool = True):
        '''calls parent app getframes method, see Cv2CaptureApplication.getframes.'''
        return self.parent_app.getframes(framecount, reset_default)

    def loop(self):
        cv2.imshow('frame', self.getframes()[0])


if __name__ == '__main__':

    appstate = Cv2CaptureApplicationState('appstate')
    app = Cv2CaptureApplication('appstate', 1, 2)
    app.add_state(appstate)
    app.main()
