import logging
import sys

# Setup structured formatter
class StructuredFormatter(logging.Formatter):
    COLOR_RESET = "\033[0m"
    COLOR_INFO = "\033[36m"    # Cyan
    COLOR_SUCCESS = "\033[32m" # Green
    COLOR_WARN = "\033[33m"    # Yellow
    COLOR_ERROR = "\033[31m"   # Red
    COLOR_DEBUG = "\033[35m"   # Magenta

    def format(self, record):
        level = record.levelname
        msg = record.getMessage()

        # Simple color mapping
        if level == "INFO":
            color = self.COLOR_INFO
        elif level == "WARNING":
            color = self.COLOR_WARN
        elif level == "ERROR":
            color = self.COLOR_ERROR
        elif level == "DEBUG":
            color = self.COLOR_DEBUG
        else:
            color = self.COLOR_RESET

        formatted = f"[HOTEL_BOT] {color}{level:<7}{self.COLOR_RESET} | {msg}"
        
        # If there's an exception, format it as well
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted

def setup_logger(name: str = "hotel_bot") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

# Shared logger instance
logger = setup_logger()
