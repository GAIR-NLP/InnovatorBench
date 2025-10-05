from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# from envs.logger import logger # TODO: Custom logger


class Source(str, Enum):
    AGENT = 'agent'
    USER = 'user'
    ENVIRONMENT = 'environment'

@dataclass
class BaseNode:
    INVALID_ID = -1

    @property
    def message(self) -> str | None:
        if hasattr(self, '_message'):
            msg = getattr(self, '_message')
            return str(msg) if msg is not None else None
        return ''

    @property
    def id(self) -> int:
        if hasattr(self, '_id'):
            id_val = getattr(self, '_id')
            return int(id_val) if id_val is not None else BaseNode.INVALID_ID
        return BaseNode.INVALID_ID

    @property
    def timestamp(self) -> str | None:
        if hasattr(self, '_timestamp') and isinstance(self._timestamp, str):
            ts = getattr(self, '_timestamp')
            return str(ts) if ts is not None else None
        return None

    @timestamp.setter
    def timestamp(self, value: datetime) -> None:
        if isinstance(value, datetime):
            self._timestamp = value.isoformat()

    @property
    def source(self) -> Source | None:
        if hasattr(self, '_source'):
            src = getattr(self, '_source')
            return Source(src) if src is not None else None
        return None
    
    @property
    def cause(self) -> int | None:
        if hasattr(self, '_cause'):
            cause_val = getattr(self, '_cause')
            return int(cause_val) if cause_val is not None else None
        return None

    @property
    def timeout(self) -> float | None:
        if hasattr(self, '_timeout'):
            timeout_val = getattr(self, '_timeout')
            return float(timeout_val) if timeout_val is not None else None
        return None
    

    def set_hard_timeout(self, value: float | None, blocking: bool = True) -> None:
        """Set the timeout for the event.

        NOTE, this is a hard timeout, meaning that the event will be blocked
        until the timeout is reached.
        """
        self._timeout = value
        if value is not None and value > 1800:
            # logger.warning(
            #     'Timeout greater than 1800 seconds may not be supported by '
            #     'the runtime. Consider setting a lower timeout.'
            # )
            print(
                'Timeout greater than 1800 seconds may not be supported by '
                'the runtime. Consider setting a lower timeout.'
            )

        # Check if .blocking is an attribute of the event
        if hasattr(self, 'blocking'):
            # .blocking needs to be set to True if .timeout is set
            self.blocking = blocking