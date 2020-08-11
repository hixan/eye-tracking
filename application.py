from pynput import keyboard
from typing import Dict, List
from collections import defaultdict
import traceback
import pyautogui
import abc


class ApplicationState:

    def __init__(self, name):
        self.binder = defaultdict(lambda: [])
        self.parent_app: Application = None
        self.name = name

    def loop(self):
        raise NotImplementedError

    def change_state(self, state):
        assert self.parent_app is not None, f'State {self.name} tried to change state when there is no parent'
        return self.parent_app.change_state(state)

    def __str__(self):
        return f'ApplicationState({self.name})'
    
    def bind(self, key, function):
        self.binder[key].append(function)


class Application:
    """Application."""

    def __init__(self, app_states: Dict[str, List[ApplicationState]], start_state: str):
        """__init__."""

        # global keybindings (default dict handles the case where there is no binding)
        self.binder = defaultdict(lambda: [], {keyboard.Key.esc: [self.exit]})

        # application states
        self.states = app_states
        self.state = start_state

        # default values
        self.mouse_movement_duration = 0.2


        self._exit = False

    @property
    def active_keys(self):
        """keys with a keybinding"""
        return self.binder.keys()

    def _onpress(self, key):
        """handles keyboard input; to pass to pynput."""
        if key in self.active_keys or True:
            print(f'{key} pressed')
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

    @abc.abstractmethod
    def frame(self):
        """Calls the correct states main loop method

        :rtype: bool
        """
        try:
            self.states[self.state].loop()
        except:
            print(f'failed in state {self.state}.')
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
        # main loop
        while True:
            self.frame()
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


if __name__ == '__main__':
    class sample_app(Application):
        def __init__(self):
             super(sample_app, self).__init__()
             self.counter = 0
        def frame(self):
            #print(f'frame number {self.counter}')
            self.counter += 1
            return True

    x = sample_app()
    x.main()


