import logging
import json
from datetime import datetime
from functools import wraps
import time
import uuid

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
custom_logger = CustomLogger()
