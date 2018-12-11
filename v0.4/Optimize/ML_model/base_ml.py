#from abc import *
from abc import ABC, abstractmethod

class Machine_Learning(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def create_ml(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def restore(self):
        pass
