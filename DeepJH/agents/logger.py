import time

from tensorboardX import SummaryWriter


class SummaryWriter_X(object):
    def __init__(self, log_dir, flush=True):
        self.logger = SummaryWriter(log_dir)
        self.flush = flush

    def add_scalar(self, name, value, step):
        self.logger.add_scalar(name, value, step)
        if self.flush:
            self._flush()

    def add_scalars(self, name, value, step):
        self.logger.add_scalars(name, value, step)
        if self.flush:
            self._flush()

    def _flush(self):
        try:
            path = self.logger.file_writer.event_writer._ev_writer._py_recordio_writer.path
            self.logger.file_writer.event_writer._ev_writer._py_recordio_writer._writer.flush()
            while True:
                if self.logger.file_writer.event_writer._event_queue.empty():
                    break
                time.sleep(0.1)  # Increased from 0.1 -> X s
            # self.logger.file_writer.event_writer._ev_writer._py_recordio_writer._writer.close()
            self.logger.file_writer.event_writer._ev_writer._py_recordio_writer._writer = open(path, 'ab')
        except:
            pass
