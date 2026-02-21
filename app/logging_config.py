"""
Structured JSON logging for production. Use setup_logging() at startup.
"""
import json
import logging


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON for aggregation and search."""

    def format(self, record: logging.LogRecord) -> str:
        log: dict = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include any extra= attributes (e.g. task_id, file) added to the LogRecord
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName", "extra",
            ) and value is not None:
                log[key] = value
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with JSON formatter to stderr."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
