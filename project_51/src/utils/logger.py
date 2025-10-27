import logging
import os
from datetime import datetime
from pathlib import Path


class OptimizationLogger:
    """
    Centralized logging system for the inventory optimization project.
    Creates date-based log folders and separate log files for each component.
    """

    def __init__(self, base_log_dir="logs"):
        """
        Initialize the logging system.

        Args:
            base_log_dir: Base directory for all logs (default: "logs")
        """
        self.base_log_dir = Path(base_log_dir)
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.log_dir = self.base_log_dir / self.today

        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize loggers
        self.loggers = {}

    def get_logger(self, component_name):
        """
        Get or create a logger for a specific component.

        Args:
            component_name: Name of the component (e.g., 'data_generation', 'rule_based', 'genetic_algorithm')

        Returns:
            Logger instance configured for the component
        """
        if component_name not in self.loggers:
            self.loggers[component_name] = self._create_logger(component_name)

        return self.loggers[component_name]

    def _create_logger(self, component_name):
        """
        Create a new logger for a component.

        Args:
            component_name: Name of the component

        Returns:
            Configured logger instance
        """
        # Create logger
        logger = logging.getLogger(f"optimization_{component_name}")
        logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()

        # Create file handler
        log_file = self.log_dir / f"{component_name}.log"
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Set formatter for handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        return logger

    def log_execution_start(self, component_name, parameters=None):
        """
        Log the start of an execution with parameters.

        Args:
            component_name: Name of the component
            parameters: Dictionary of parameters used in the execution
        """
        logger = self.get_logger(component_name)

        logger.info("=" * 60)
        logger.info("=" * 60)
        logger.info(f"Starting {component_name.replace('_', ' ').title()} Execution")
        logger.info("=" * 60)

        if parameters:
            logger.info("Execution Parameters:")
            for key, value in parameters.items():
                logger.info(f"  - {key}: {value}")

        logger.info(f"Log file: {self.log_dir / f'{component_name}.log'}")
        logger.info("=" * 60)

    def log_execution_end(self, component_name, execution_time=None, results=None):
        """
        Log the end of an execution with results.

        Args:
            component_name: Name of the component
            execution_time: Total execution time in seconds
            results: Dictionary of execution results
        """
        logger = self.get_logger(component_name)

        logger.info("-" * 60)
        logger.info(f"{component_name.replace('_', ' ').title()} Execution Completed")

        if execution_time:
            logger.info(f"Total Execution Time: {execution_time:.2f} seconds")

        if results:
            logger.info("Execution Results:")
            for key, value in results.items():
                logger.info(f"  - {key}: {value}")

        logger.info("=" * 60)

    def log_progress(self, component_name, message):
        """
        Log a progress message.

        Args:
            component_name: Name of the component
            message: Progress message to log
        """
        logger = self.get_logger(component_name)
        logger.info(message)

    def log_error(self, component_name, error_message, exception=None):
        """
        Log an error message.

        Args:
            component_name: Name of the component
            error_message: Error message to log
            exception: Exception object (optional)
        """
        logger = self.get_logger(component_name)

        if exception:
            logger.error(f"ERROR: {error_message}")
            logger.error(f"Exception: {str(exception)}")
        else:
            logger.error(f"ERROR: {error_message}")

    def log_warning(self, component_name, warning_message):
        """
        Log a warning message.

        Args:
            component_name: Name of the component
            warning_message: Warning message to log
        """
        logger = self.get_logger(component_name)
        logger.warning(f"WARNING: {warning_message}")

    def get_log_summary(self):
        """
        Get a summary of all log files created today.

        Returns:
            Dictionary with log file information
        """
        log_files = []

        if self.log_dir.exists():
            for log_file in self.log_dir.glob("*.log"):
                file_size = log_file.stat().st_size
                log_files.append(
                    {
                        "file": log_file.name,
                        "path": str(log_file),
                        "size_bytes": file_size,
                        "size_mb": file_size / (1024 * 1024),
                    }
                )

        return {
            "date": self.today,
            "log_directory": str(self.log_dir),
            "log_files": log_files,
            "total_files": len(log_files),
        }


# Convenience function to get a logger instance
def get_optimization_logger(base_log_dir="logs"):
    """
    Get a logger instance for the optimization system.

    Args:
        base_log_dir: Base directory for logs

    Returns:
        OptimizationLogger instance
    """
    return OptimizationLogger(base_log_dir)
