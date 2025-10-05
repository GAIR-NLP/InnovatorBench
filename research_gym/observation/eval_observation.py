from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Type
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum

from research_gym.base_node import BaseNode
from research_gym.schema.action import ActionType
from research_gym.schema.observation import ObservationType
from research_gym.action import BaseAction
from research_gym.observation.base_observation import BaseObservation


@dataclass
class EvalObservation(BaseObservation):
    score: int = 0
    task_name: str = "Default"           # Task name
    eval_results: dict = None     # Evaluation results
    already_eval_num: int = 0
    
    
    def __post_init__(self):
        super().__init__()
        self._observation_type = ObservationType.EVAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = super().to_dict()
        result.update({
            # 'task_name': self.task_name,
            'overall_score': self.score,
            "already_eval_num": self.already_eval_num
        })
        
        # TODO: self.eval_results currently only has metric_0, may be modified later
        for key, value in self.eval_results.items():
            result[key] = value

        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_eval_result(cls, result: Dict[str, Any], action: BaseAction) -> 'EvalObservation':
        """Create EvalObservation from evaluation result"""
        observation = cls()
        observation.tool_call_id = action.call_id
        assert type(action.action_type) == ActionType
        observation.tool_name = action.action_type
        # observation.task_name = result["task_name"]
        observation.score = result["score"]
        observation.eval_results = result["eval_results"]
        observation.already_eval_num = result["already_eval_num"]
        observation.success = result.get('success', False)
        observation._message = result.get('message') if observation.success and result.get('message') else None
        observation.error_message = result.get('message') if not observation.success else None
        return observation
    
    def __str__(self) -> str:
        ret = f'**EvalObservation**\n'
        ret += self._get_base_info()
        if self.task_name:
            ret += f'Task name: {self.task_name}\n'
        if self.score:
            ret += f'Score: {self.score}\n'
        if self.eval_results:
            ret += f'Eval results: {self.eval_results}\n'
        return ret