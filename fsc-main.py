import multiprocessing
from multiprocessing import Manager

from workers.recorder import record_audio
from workers.extractor import feature_extractor


if __name__ == "__main__":

    manager = Manager()
    audioIndex = manager.list()
    features = manager.list()

    p1 = multiprocessing.Process(target=record_audio, args=[audioIndex])
    p2 = multiprocessing.Process(target=feature_extractor, args=[audioIndex,features])

    p1.start()
    p2.start()

    p1.join()
    p2.join()

