# src/utils/logger.py

"""
Logger Utility
--------------
Provides a centralized logging utility using loguru.
Ensures consistent, structured logs across training,
evaluation, API, and dashboard.
"""

from loguru import logger
import sys


# Configure logger
logger.remove()  # remove default
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True,  # useful for debugging
    level="INFO"
)

# Example usage:
# from utils.logger import logger
# logger.info("Training started")
# logger.warning("Learning rate too high?")
# logger.error("Model crashed")
