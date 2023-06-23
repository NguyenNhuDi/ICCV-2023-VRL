import logging
import multiprocessing as mp
import queue
import time

class LogHelper:
    def __init__(self, out:str) -> None:
        self.log_messages = mp.JoinableQueue()
        self.logger = mp.Process(target=LogHelper.logger_process, args=(out, self.log_messages,))
        self.__start__()

    def __start__(self):
        self.logger.start()

    def join(self):
        self.logger.join()

    def _print(self, *args):
        for m in args:
            self.log_messages.put(m)

    @staticmethod
    def logger_process(out:str, log_messages: mp.JoinableQueue):
        logging.basicConfig(filename=out,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level='DEBUG')
         
        logger = logging.getLogger()

        while True:
            try:
                message = log_messages.get(1)
                if message is None:
                    break
                logger.error(message)
            except queue.Empty:
                time.sleep(1)  # Sleep for a while before trying again.
                print('Empty')
                continue
            else:
                log_messages.task_done()