from pynput import keyboard
import pyautogui
import abc


class Application:
    """Application."""

    def __init__(self):
        """__init__."""
        # define defaults
        self.positive_action_keys = {keyboard.Key.space}
        self.exit_action_keys = {keyboard.Key.esc}
        self.mouse_movement_duration = .2
        self._exit = False

    @property
    def active_keys(self):
        """active_keys."""
        return self.positive_action_keys | self.exit_action_keys

    def _onpress(self, key):
        """handles keyboard input; to pass to pynput."""
        if key in self.active_keys:
            print(f'{key} pressed')
        if key in self.positive_action_keys:
            self.positive_action()
        elif key in self.exit_action_keys:
            self.exit_action()

    def exit_action(self):
        """trigger exiting the application."""
        self._exit = True

    def positive_action(self):
        """normally triggers a snapshot"""
        print('No positive action behaviour defined.')

    @abc.abstractmethod
    def frame(self) -> bool:
        """frame. Returns true if frame was calculated with no errors

        :rtype: bool
        """

        raise NotImplementedError

    def indicate_location(self, x, y):
        """indicate_location.

        :param x: x coordinate
        :param y: y coordinate
        """

        pyautogui.moveTo(x, y, self.mouse_movement_duration)
        return pyautogui.position()
        
    def main(self):
        """main."""
        # start keyboard listener
        listener = keyboard.Listener(on_press=self._onpress)
        listener.start()
        # main loop
        while self.frame():
            if self._exit:
                break
        listener.stop()



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


