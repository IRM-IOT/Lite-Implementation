import multiprocessing
from multiprocessing import Manager

from workers.recorder import record_audio


if __name__ == "__main__":

    manager = Manager()
    audioIndex = manager.list()
    features = manager.list()

    p1 = multiprocessing.Process(target=record_audio, args=[audioIndex])

    p1.start()

    p1.join()

