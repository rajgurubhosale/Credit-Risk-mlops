import sys
from src.logger import config_logger


class MyException(Exception):
    """
    Custom Exception class for handling error in the HomeCredit Risk System
    """

    def __init__(self,error:str , error_details :sys,logger:object):
        """
        
        params:
            error : A String describing the error
            error_details: Sys module to extract the error_details
            logger: logger object  to log the errors and details
        """

        super().__init__(error)
        
        error_type,_,error_info = error_details.exc_info()

        #extract the line_number from the error details where error occuered
        line_number = error_info.tb_lineno

        #extract the file_name from the error details where error occuered
        file_name = error_info.tb_frame.f_code.co_filename
        
        #extract type of error
        error_type = error_type.__name__

        #create the custom foramatted message for the error to simplify debugging
        self.error_message = f"Error Occured in python script:[{file_name}] at line number [{line_number}] : {error_type}: {str(error)}"

        #logs the error message
        logger.error(self.error_message)

    def __str__(self):

        return self.error_message