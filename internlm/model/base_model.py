from abc import ABCMeta, abstractmethod

from torch import nn


class BaseModel(nn.Module, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def load_hf_weights(folder: str, model: nn.Module) -> None:
        raise NotImplementedError
