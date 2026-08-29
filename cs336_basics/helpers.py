from contextlib import contextmanager
import time

@contextmanager
def timer(name: str, disable: bool = False):
    if disable:
        yield
        return
    
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"Timing info: \t{name} \t{end - start:.6f}s")