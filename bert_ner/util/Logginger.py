#encoding:utf-8
import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler

'''
使用方式
from you_logging_filename.py import init_logger
logger = init_logger("dataset",logging_path='')
def you_function():
	logger.info()
	logger.error()

'''


'''
日志模块
1. 同时将日志打印到屏幕跟文件中
2. 默认值保留近7天日志文件
'''
def init_logger(logger_name, logging_path):
    if logger_name not in Logger.manager.loggerDict:
        logger  = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler(filename=logging_path+"/all.log",when='D',backupCount = 7)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str,datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        console= logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        handler = TimedRotatingFileHandler(filename=logging_path+"/error.log",when='D',backupCount=7)
        datefmt = '%Y-%m-%d %H:%M:%S'
        format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
        formatter = logging.Formatter(format_str,datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    logger = logging.getLogger(logger_name)
    return logger

#if __name__ == "__main__":
#     logger = init_logger("datatest",logging_path="E:/neo4j-community-3.4.1")
#     logger.error('test_error')
#     logger.info("test-info")
#     logger.warn("test-warn")
	 
	 
	 