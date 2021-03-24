import logging
import random
import time
from multiprocessing import Process
from logging import handlers

import zmq
from zmq.log.handlers import PUBHandler


class LogPublisher(object):
    """
    Centralized logger publisher, used in MP.Process.run() to send log message to log collector.
    """

    def __init__(self, ip="127.0.0.1", port=8000):
        self._logger = logging.getLogger("LogPublisher")
        self._logger.setLevel(logging.DEBUG)
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind("tcp://{}:{}".format(ip, port))
        self.handler = PUBHandler(self.socket)
        self.format = logging.Formatter("[%(filename)s:%(lineno)d] %(levelname)s %(message)s")
        self.handler.setFormatter(self.format)
        self._logger.addHandler(self.handler)

    @property
    def logger(self):
        return self._logger


class LogCollector(Process):
    """
    Centralized logger processor, a process used to collected log message sent by other processes.
    """

    def __init__(self, port=8000):
        super().__init__()
        self.port = port

    def run(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:{}".format(self.port))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.logger = logging.getLogger("LogCollector")
        self.logger.setLevel(logging.DEBUG)
        file_handler = handlers.RotatingFileHandler("./nano.log", maxBytes=10*1024*1024, backupCount=5)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        while True:
            topic, msg = self.socket.recv_multipart()
            self.logger.log(getattr(logging, topic.decode()), msg.decode().strip())


class Publisher(Process):

    def __init__(self):
        super().__init__()
        self.logger = None

    def run(self):
        self.logger = LogPublisher().logger
        time.sleep(1)

        while True:
            topic = random.randrange(9999, 10005)
            msg = random.randrange(1, 255) - 80
            self.logger.info("{}:{}".format(topic, msg))


if __name__ == '__main__':
    collector = LogCollector()
    collector.start()

    publish = Publisher()
    publish.start()

    collector.join()
    publish.join()



