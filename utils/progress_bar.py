import time
from datetime import datetime, timedelta
from math import ceil, sqrt
from random import random, randint
from .ansi import Ansi


class Loader:
    def __init__(self,
                 name: str = "Loading",
                 loader_type: str = "slash",  # in ["slash", "pie", "quarter", "snake", "running_square", "corner", "points"]
                 loader_color: Ansi = Ansi.cyan,):

        self.name = name
        self.loader_type = loader_type
        self.loader_color = loader_color
        self.done = False

        self.loader_sequence = ["/", "-", "\\", "|"]
        if self.loader_type == "pie":
            self.loader_sequence = ["○", "◔", "◑", "◕", "●"]
        if self.loader_type == "quarter":
            self.loader_sequence = ["◴", "◷", "◶", "◵"]
        if self.loader_type == "snake":
            self.loader_sequence = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        if self.loader_type == "running_square":
            self.loader_sequence = ["■□□□", "□■□□", "□□■□", "□□□■"]
        if self.loader_type == "corner":
            self.loader_sequence = ["▟", "▙", "▛", "▜"]
        if self.loader_type == "points":
            self.loader_sequence = [".", "..", "...", "...."]
        self.loader_sequence_position = 0

    def step(self):
        print(self, end="\r")
        self.loader_sequence_position += 1

    def __str__(self):
        if self.done:
            return self.name + " DONE."
        return self.name + " " + self.loader_sequence[self.loader_sequence_position % len(self.loader_sequence)]


class ProgressBar:
    def __init__(self,
                 name: str = "",
                 max_value=100,
                 min_value=0,
                 step=1,
                 print_length=None,
                 name_print_length=None,
                 reverse=False,
                 estimate_duration=None,
                 show_bar_name: bool = True,
                 bar_name_color: Ansi = Ansi.yellow,
                 show_bar: bool = True,
                 loaded_bar_color: Ansi = Ansi.green,
                 unloaded_bar_color: Ansi = Ansi.grey,
                 show_percent: bool = True,
                 percent_color: Ansi = Ansi.purple_deep,
                 show_time_left: bool = True,
                 time_left_color: Ansi = Ansi.blue,
                 ):

        if estimate_duration is None:
            estimate_duration = show_time_left

        self.name = name
        self.max_value = max_value
        self.min_value = min_value
        self.step_size = step
        self.print_length = print_length
        self.name_print_length = name_print_length
        self.reverse = reverse
        self.estimate_duration = estimate_duration
        self.show_bar_name = show_bar_name
        self.bar_name_color: Ansi = bar_name_color
        self.show_bar = show_bar
        self.loaded_bar_color: Ansi = loaded_bar_color
        self.unloaded_bar_color: Ansi = unloaded_bar_color
        self.show_percent = show_percent
        self.percent_color: Ansi = percent_color
        self.show_time_left = show_time_left
        self.time_left_color: Ansi = time_left_color
        self.minimal_size = self.get_minimal_size()

        self.n = min_value

        if estimate_duration:
            self.last_step = datetime.now()
            self.time_per_step = None
            self.estimated_end = None

    def percent(self):
        start = self.min_value
        stop = self.max_value

        return (self.n - start) / (stop - start) * 100

    def step(self, inc: int = None):
        if inc is None:
            inc = self.step_size
        self.n = min(self.max_value, self.n + inc)

        if self.estimate_duration:
            last_step = self.last_step
            self.last_step = datetime.now()
            t = (self.last_step - last_step) / inc
            lr = 1 / sqrt(self.n / self.max_value * 100)
            if self.time_per_step is None:
                self.time_per_step = t
            else:
                self.time_per_step = self.time_per_step * (1 - lr) + t * lr
            self.estimated_end = self.last_step + self.time_per_step * (self.max_value - self.n)

    def __str__(self):
        percent = self.percent()
        if self.reverse:
            percent = 100 - percent

        result = ""
        if self.show_bar_name:
            name = self.name
            if self.name_print_length is not None:
                if len(self.name) > self.name_print_length:
                    name = self.name[:self.name_print_length]
                else:
                    name = self.name + " " * (self.name_print_length - len(self.name))
            result += self.bar_name_color + name + ": "
        if self.show_bar:
            bar_length = (self.print_length - self.minimal_size) if self.print_length is not None else 100
            unloaded_length = ceil(bar_length * (1 - percent / 100))
            loaded_length = bar_length - unloaded_length
            result += self.unloaded_bar_color + "[" \
            + self.loaded_bar_color + "━" * loaded_length \
            + self.unloaded_bar_color + "━" * unloaded_length + "]  "
        if self.show_percent:
            msg = str(ceil(percent))
            result += self.percent_color + " " * (3 - len(msg)) + msg + "%"
        if self.show_time_left:
            if self.estimated_end is None:
                result += self.time_left_color + " ..:..:.."
            else:
                now = datetime.now()
                time_left: timedelta = self.estimated_end - now
                if time_left.days < 0:
                    time_left = timedelta()

                hours, remainder = divmod(time_left.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                result += " " + self.time_left_color + '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))
        result += Ansi.NONE
        return result

    def get_minimal_size(self):
        minimal_size = 0
        minimal_size += 3 if self.show_bar else 0
        if self.name_print_length is None:
            minimal_size += len(self.name) + 1 if self.show_bar_name else 0
        else:
            minimal_size += self.name_print_length + 1 if self.show_bar_name else 0
        minimal_size += 6 if self.show_percent else 0  # " " * 2 + "100%"
        minimal_size += 9 if self.show_time_left else 0  # " " + "00:00:00"
        return minimal_size

    def reset(self):
        self.n = self.min_value

        # Recompute minimal_size
        self.minimal_size = self.get_minimal_size()


if __name__ == "__main__":

    loader = Loader("Initialising usage examples ...", "snake")
    for _ in range(10):
        loader.step()
        time.sleep(.4)
    print("")

    bar = ProgressBar(name="Loading ... ", reverse=False, print_length=100)
    for i in range(50):
        bar.step(randint(1, 4))
        time.sleep(random() / 2)
    print("")

    bar.name = "Unloading ..."
    bar.reverse = True
    bar.reset()
    for i in range(50):
        bar.step(randint(1, 4))
        time.sleep(random() / 2)
    print("")

    bar.name = "I have be reset."
    bar.reverse = False
    bar.reset()
    for i in range(50):
        bar.step(randint(1, 4))
        time.sleep(0.05)
    print("")

    bar.name = "Where bar is ?"
    bar.show_bar = False
    bar.reset()
    for i in range(50):
        bar.step(randint(1, 4))
        time.sleep(0.1)
    print("")

