import multiprocessing


def init_global_variables():
    global stop_flag
    stop_flag = False
    global mylist
    mylist = []


manager = multiprocessing.Manager()
value = manager.Value(bool, False)
