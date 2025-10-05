"""
ObservationType to Observation Class Mapping
Provide mapping relationship from ObservationType to specific Observation classes
"""

from typing import Dict, Type, Optional

from research_gym.schema.observation import ObservationType
from research_gym.observation.base_observation import BaseObservation

# Import all Observation classes
from research_gym.observation.command_observation import (
    CommandObservation,
    SessionCreateCommandObservation,
    ListCommandObservation,
    RunCommandObservation,
    SessionInputCommandObservation,
    SessionOutputCommandObservation,
    SessionStatusCommandObservation,
    SessionClearCommandObservation,
    SessionCloseCommandObservation,
    SessionKillCommandObservation,
)

from research_gym.observation.file_observation import (
    FileObservation,
    ReadFileObservation,
    WriteFileObservation,
    SearchFileDirObservation,
    ListFileObservation,
    FileInfoObservation,
    ErrorObservation,
)

from research_gym.observation.parse_observation import ParseObservation
from research_gym.observation.eval_observation import EvalObservation
from research_gym.observation.search_observation import SearchObservation
from research_gym.observation.web_browse_observation import (
    WebBrowseObservation,
    ReadWebObservation,
    SearchWebObservation,
    ListLinksObservation,
)


# ObservationType to Observation class mapping table
OBSERVATION_TYPE_TO_CLASS: Dict[ObservationType, Type[BaseObservation]] = {
    # Basic
    ObservationType.BASE: BaseObservation,

    # Command related
    ObservationType.COMMAND: CommandObservation,
    ObservationType.COMMAND_CREATE_SESSION: SessionCreateCommandObservation,
    ObservationType.COMMAND_LIST_SESSIONS: ListCommandObservation,
    ObservationType.COMMAND_RUN: RunCommandObservation,
    ObservationType.COMMAND_INPUT_IN_SESSION: SessionInputCommandObservation,
    ObservationType.COMMAND_OUTPUT_SESSION: SessionOutputCommandObservation,
    ObservationType.COMMAND_SESSION_STATUS: SessionStatusCommandObservation,
    ObservationType.COMMAND_CLEAR_SESSION_BUFFER: SessionClearCommandObservation,
    ObservationType.COMMAND_CLOSE_SESSION: SessionCloseCommandObservation,
    ObservationType.COMMAND_KILL_PROCESSES: SessionKillCommandObservation,

    # File related
    ObservationType.FILE: FileObservation,
    ObservationType.FILE_READ: ReadFileObservation,
    ObservationType.FILE_WRITE: WriteFileObservation,
    ObservationType.FILE_SEARCH: SearchFileDirObservation,
    ObservationType.FILE_LIST: ListFileObservation,
    ObservationType.FILE_INFO: FileInfoObservation,
    ObservationType.FILE_ERROR: ErrorObservation,

    # Parse, evaluation, search, web
    ObservationType.PARSE: ParseObservation,
    ObservationType.EVAL: EvalObservation,
    ObservationType.WEB_SEARCH: SearchObservation,
    ObservationType.WEB_BROWSE: WebBrowseObservation,
    ObservationType.WEB_BROWSE_READ: ReadWebObservation,
    ObservationType.WEB_BROWSE_SEARCH: SearchWebObservation,
    ObservationType.WEB_BROWSE_LINKS: ListLinksObservation,
}


# Reverse mapping: Observation class to ObservationType
CLASS_TO_OBSERVATION_TYPE: Dict[Type[BaseObservation], ObservationType] = {
    obs_class: obs_type for obs_type, obs_class in OBSERVATION_TYPE_TO_CLASS.items()
}


def get_observation_class(observation_type: ObservationType) -> Optional[Type[BaseObservation]]:
    """
    Get the corresponding Observation class according to ObservationType
    """
    return OBSERVATION_TYPE_TO_CLASS.get(observation_type)


def get_observation_type(observation_class: Type[BaseObservation]) -> Optional[ObservationType]:
    """
    Get the corresponding ObservationType according to Observation class
    """
    return CLASS_TO_OBSERVATION_TYPE.get(observation_class)


def get_supported_observation_types() -> list[ObservationType]:
    """Get all supported ObservationType lists"""
    return list(OBSERVATION_TYPE_TO_CLASS.keys())


def get_observation_info(observation_type: ObservationType) -> Optional[Dict[str, str]]:
    """
    Get basic information of Observation class
    """
    obs_class = get_observation_class(observation_type)
    if obs_class:
        return {
            'observation_type': observation_type.value,
            'class_name': obs_class.__name__,
            'module': obs_class.__module__,
            'description': getattr(obs_class, '__doc__', '') or '',
        }
    return None


def print_mapping_table():
    """Print the complete mapping table for debugging"""
    print("ObservationType to Observation Class Mapping:")
    print("=" * 50)
    for obs_type, obs_class in sorted(OBSERVATION_TYPE_TO_CLASS.items(), key=lambda x: x[0].value):
        print(f"{obs_type.value:<30} -> {obs_class.__name__}")
    print("=" * 50)
    print(f"Total mappings: {len(OBSERVATION_TYPE_TO_CLASS)}")

def create_observation_by_type(observation_type: ObservationType, **kwargs) -> Optional[BaseObservation]:
    """
    Create the corresponding Observation instance according to ObservationType
    """
    obs_class = get_observation_class(observation_type)
    return obs_class(**kwargs) if obs_class else None

if __name__ == "__main__":
    print_mapping_table()

