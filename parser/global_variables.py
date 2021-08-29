import multiprocessing


def _init():
    global global_dict
    _global_dict = {}
    # global stop_flag
    # stop_flag = False
    # global mylist
    # mylist = []
    # global avg
    # avg = avg_matrixes(2)


def set_value(key, value):
    _global_dict[key] = value


def get_value(key):
    return _global_dict[key]


multiprocessing.freeze_support()

manager = multiprocessing.Manager()
value = manager.Value(bool, False)
