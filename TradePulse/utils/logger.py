"""
Structured logging setup for TradePulse components.
"""
import logging
import sys
import os
from datetime import datetime

def setup_logger(name, log_level=logging.INFO, log_to_file=True):
    """
    Set up a configured logger for the specified component.
    
    Args:
        name (str): Name for the logger, typically the module name
        log_level (int): Logging level (default: logging.INFO)
        log_to_file (bool): Whether to save logs to file (default: True)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

if __name__ == "__main__":
    # Example usage
    logger = setup_logger("test_logger")
    logger.info("This is an info message")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.debug("This is a debug message (not shown with default INFO level)")
    
    # Change log level to show debug messages
    debug_logger = setup_logger("debug_test", log_level=logging.DEBUG)
    debug_logger.debug("This debug message will be shown") 