# coding=utf-8


class SimpleLogger(object):
    def __init__(self, logfile, terminal):
        self.log = open(logfile, "a")
        self.terminal = terminal
        self.isatty = lambda: False

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
