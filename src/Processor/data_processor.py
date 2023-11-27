from abc import ABC, abstractmethod

import pandas as pd


class DataProcessor(ABC):
    def __init__(self):
        """
        Base class for processing neuron data. Configs the radius of interest.
        """
        pass

    @abstractmethod
    def extract_synapse(self):
        """
        Abstract method to extract a specific synapse (usually by location)
        and return the synapse (usually as a NetworkX graph) along with the
        corresponding surrounding mesh.
        """
        pass

    @abstractmethod
    def filter_synapse(self) -> bool:
        """
        Abstract method to take a measured synapses and determine if it is valid.

        Filters out the "bad" data.
        """
        pass

    @abstractmethod
    def measure_head_to_neck_ratio(self) -> float:
        """
        Abstract method to, given a synapse, measure its head to neck ratio.
        """
        pass

    @abstractmethod
    def measure_all_synapses(self):
        """
        Abstract method to measure all synapses head/neck ratios from a neuron.
        """
        pass
