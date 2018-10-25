import logging
import sys

log = None
_initialized_flag = False


def setup():
    global log, _initialized_flag

    if _initialized_flag:
        return

    # create logger
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    # create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s  %(message)s',
    #                               datefmt='%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('[%(asctime)s]-[%(levelname)s]-[%(module)s]-[%(funcName)s] %(message)s',
    #                               datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d|%(levelname)s|%(module)s.%(funcName)s| %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s %(message)s',
    #                               datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)

    # add the handlers to logger
    log.addHandler(console_handler)

    log.info('Initializing logs.')
    _initialized_flag = True


setup()
