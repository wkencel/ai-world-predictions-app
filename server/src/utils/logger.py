import logging
from datetime import datetime
import json
from functools import wraps
import time
import uuid
from typing import Any, Dict
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform color support
init()

class ColorLogger:
    def __init__(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PredictionFramework')

    def info(self, message: str):
        self.logger.info(f"{Fore.BLUE}{message}{Style.RESET_ALL}")

    def success(self, message: str):
        self.logger.info(f"{Fore.GREEN}{message}{Style.RESET_ALL}")

    def error(self, message: str):
        self.logger.error(f"{Fore.RED}{message}{Style.RESET_ALL}")

    def warning(self, message: str):
        self.logger.warning(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")

    def prediction(self, message: str):
        self.logger.info(f"{Fore.MAGENTA}🎯 {message}{Style.RESET_ALL}")

    def expert(self, expert_name: str, message: str):
        self.logger.info(f"{Fore.CYAN}👨‍🔬 {expert_name}: {message}{Style.RESET_ALL}")

    def analysis(self, message: str):
        self.logger.info(f"{Fore.WHITE}📊 {message}{Style.RESET_ALL}")

    def json_log(self, data: Any, prefix: str = ""):
        """Pretty print JSON data"""
        try:
            formatted_json = json.dumps(data, indent=2)
            self.logger.info(f"{Fore.WHITE}{prefix}\n{formatted_json}{Style.RESET_ALL}")
        except Exception as e:
            self.error(f"Error formatting JSON: {str(e)}")

    def log_service_call(self, service_name):
        """Decorator for logging service calls"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                request_id = str(uuid.uuid4())
                start_time = time.time()

                self.info(f"Starting {service_name} call - {func.__name__}")

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    self.success(f"Completed {service_name} call - {func.__name__} ({execution_time:.2f}s)")
                    return result

                except Exception as e:
                    self.error(f"Failed {service_name} call - {func.__name__}: {str(e)}")
                    raise

            return wrapper
        return decorator

    def server_start(self):
        """Log server start with a distinctive separator"""
        separator = f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}"
        self.logger.info(f"{separator}\n🚀 SERVER STARTING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{separator}")

class ColorFormatter(logging.Formatter):
    """Custom formatter for colored logs"""

    def format(self, record):
        # Add timestamp to all logs
        timestamp = datetime.now().strftime('%H:%M:%S')

        if record.levelno == logging.ERROR:
            prefix = f"{Fore.RED}[{timestamp}]{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            prefix = f"{Fore.YELLOW}[{timestamp}]{Style.RESET_ALL}"
        else:
            prefix = f"{Fore.CYAN}[{timestamp}]{Style.RESET_ALL}"

        return f"{prefix} {record.getMessage()}"

class CustomLogger:
    def __init__(self):
        self.logger = logging.getLogger('ai-predictions')
        self.logger.setLevel(logging.INFO)

        # Add file handler
        fh = logging.FileHandler('predictions.log')
        fh.setLevel(logging.INFO)

        # Add console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_service_call(self, service_name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                request_id = str(uuid.uuid4())
                start_time = time.time()

                self.logger.info(json.dumps({
                    'request_id': request_id,
                    'service': service_name,
                    'function': func.__name__,
                    'status': 'started',
                    'timestamp': datetime.now().isoformat(),
                    'args': str(args),
                    'kwargs': str(kwargs)
                }))

                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    self.logger.info(json.dumps({
                        'request_id': request_id,
                        'service': service_name,
                        'function': func.__name__,
                        'status': 'completed',
                        'execution_time': execution_time,
                        'timestamp': datetime.now().isoformat()
                    }))
                    return result

                except Exception as e:
                    self.logger.error(json.dumps({
                        'request_id': request_id,
                        'service': service_name,
                        'function': func.__name__,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }))
                    raise

            return wrapper
        return decorator

# Create a singleton instance
color_logger = ColorLogger()
