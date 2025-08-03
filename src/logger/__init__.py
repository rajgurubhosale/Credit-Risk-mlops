import os 
import logging
from from_root import from_root
from logging.handlers import RotatingFileHandler

#constant for logging
LOG_DIR ='logs'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3
LOG_DIR_PATH =  os.path.join(from_root(),LOG_DIR)


#check if LOG_DIR_PATH exist if not then create the dir
os.makedirs(LOG_DIR_PATH,exist_ok=True)

def config_logger(logger_name:str)->object:
    """
    config  and create the logger with rotatating file_handler and console_handler with format.
    
    args:
        logger_name: name to define the log file name in the dir logs.
    
    return:
        logger: object of configuered logger.
    """
   
    #custom logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    #creating the loger name log file 
    logger_file_path = os.path.join(LOG_DIR_PATH,logger_name+'.log')
    # File handler with the rotating handler
    file_handler = RotatingFileHandler(logger_file_path,maxBytes=MAX_LOG_SIZE,backupCount=BACKUP_COUNT)
    file_handler.setLevel(logging.DEBUG)

    #Console Handler
    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.INFO)

    #creating custom logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    #set Format to the handlers
    file_handler.setFormatter(formatter)
    consol_handler.setFormatter(formatter)

    #add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(consol_handler)
    

    return logger