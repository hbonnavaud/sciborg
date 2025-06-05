import time
from copy import deepcopy


class Stopwatch:
    """
    A stopwatch that can be build using any clock. The function that will be used to get the current time
    """
    def __init__(self, get_time=time.process_time):
        assert callable(get_time)
        temp_time = time.process_time()
        self.duration = temp_time - temp_time  # Initialise to a duration of 0.
        self.last_start = None
        self.get_time = get_time

    def start(self):
        assert not self.is_running(), "Cannot start a running stopwatch."
        self.last_start = self.get_time()

    def stop(self):
        assert self.is_running(), "Stopwatch has not been started. There is probably an error."
        self.duration += self.get_time() - self.last_start
        self.last_start = None
        return self.duration

    def get_duration(self):
        if self.is_running():
            return self.duration + (self.get_time() - self.last_start)
        else:
            return deepcopy(self.duration)

    def is_running(self):
        return self.last_start is not None

    def reset(self, start=False):
        self.__init__()
        if start:
            self.start()

    def __str__(self):
        return str(self.get_duration())


if __name__ == "__main__":
    print("cpu time: ", time.process_time())

    sw = Stopwatch()
    sw.start()
    print("stated ..")
    for i in range(10000):  # Random operations to take some time. I can use a sleep() but I did like this idkw.
        a = i ** i
    sw.stop()
    print("stopped.")
    print("duration: " + str(sw) + " seconds")