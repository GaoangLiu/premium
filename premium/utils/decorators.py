import codefast as cf
import time
from functools import wraps


def runninglog(func):
    def wrapper(*args, **kwargs):
        cf.info('{}() started'.format(func.__name__))
        res = func(*args, **kwargs)
        cf.info('{}() ended'.format(func.__name__))
        return res

    return wrapper


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        cf.info(f'{func.__name__} took {end - start} seconds')
        return res

    return wrapper