from dataclasses import dataclass
from datetime import datetime
import enum
from typing import List, Dict, Any, Optional

class ComputerType(enum.Enum):
    GPU = "gpu"
    LOCALHOST_CPU = "localhost_cpu"
    CPU = "cpu"

@dataclass
class ComputerConfig:
    computer_ip: str
    computer_port: int
    computer_type: ComputerType
    available: bool

@dataclass
class TaskConfig:
    task_name: str
    workspace: str
    tokenizer: str
    max_working_time: float
    start_time: datetime
    max_steps: int
    max_eval_num: int
    
    # train
    model_name: str
    checkpoint_base_path: str  # Path should not end with "/"
    workspace_dataset_path: str  # Path to the dataset of the workspace
    actual_workspace: str  # Real path of the symbolic link
    resume_from_path: str  # Resume from which checkpoint
    save_freq: int  # Save frequency

    env_vars: Dict[str, str]
    cmd_proxy_url: str  # Renamed: proxy URL dedicated to command operations
    computer_pool: List[ComputerConfig]
    launch_type: str

    # docker
    default_shell: str
    default_http_port: int
    
    # openai
    openai_api_key: str
    openai_base_url: str

    # evaluation
    # agent_output_dir: str
    # reference_dir: str
    eval_workspace: str
    max_eval_time: float
    
    # search engine configuration
    search_engine: str = ""  # Search engine type: 'google', 'bing', or empty string to disable
    serper_api_key: str = ""  # Google search API key
    azure_bing_search_subscription_key: str = ""  # Bing search API key
    search_max_top_k: int = 100  # Maximum number of search results
    search_region: str = "us"  # Search region
    search_lang: str = "en"  # Search language
    azure_bing_search_mkt: str = "en-US"  # Bing search market
    search_cache_dir: str = "./search_cache"  # Search cache directory
    search_cache_duration_days: int = 1  # Search cache duration in days

    # web browsing configuration
    web_server_host: str = "localhost"  # Web server host
    web_server_port: int = 8124  # Web server port
    web_proxy_url: str = ""  # Proxy URL dedicated to web operations
    web_cache_dir: str = "./web_cache"  # Web cache directory
    web_cache_duration_days: int = 1  # Web cache duration in days

    
    # Backward compatibility support
    @property
    def proxy_url(self) -> str:
        """Backward compatibility property, returns cmd_proxy_url"""
        return self.cmd_proxy_url