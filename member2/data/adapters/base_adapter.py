from abc import ABC, abstractmethod
from investigation.alert_schema import SecurityAlert


class BaseDatasetAdapter(ABC):
    """
    Base class for converting different IDS datasets
    into the common SecurityAlert format.
    """

    @abstractmethod
    def convert(self, row) -> SecurityAlert:
        """
        Convert one dataset row into a SecurityAlert.
        """
        pass