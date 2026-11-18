"""A wall-clock limit for running engine-written code in this process."""
from contextlib import contextmanager
import signal
import threading


class Timeout(BaseException):
    """Not an Exception on purpose: operators that catch Exception must not swallow it."""


@contextmanager
def time_limit(seconds):
    """Stop a composed program that runs away.

    Exhaustive search drops oversized intermediate values before composing further; a program
    executed as a whole on new arguments has no such bound, and an operator's guard is not part
    of the code it emits.
    """
    if threading.current_thread() is not threading.main_thread():
        yield                       # signals only reach the main thread; run unguarded elsewhere
        return

    def interrupt(signum, frame):
        raise Timeout

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
