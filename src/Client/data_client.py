from abc import ABC, abstractmethod

import pandas as pd

from src.Neuron.neuron import Neuron


class DataClient(ABC):
    def __init__(self):
        """
        Base class for fetching synapse data from a given CloudVolume source.
        """
        pass

    @abstractmethod
    def change_cloud_uri(self, *args, **kwargs) -> None:
        """
        Abstract method for changing a CloudVolume data source.
        """
        pass

    @abstractmethod
    def change_cave_uri(self, *args, **kwargs) -> None:
        """
        Abstract method for changing a CAVEclient data source.
        """
        pass

    @abstractmethod
    def get_proofread_neurons(self, *args, **kwargs):
        """
        Abstract method to get proofread neurons from a data source and store them.
        """
        pass

    @abstractmethod
    def skeletonize(self, *args, **kwargs):
        """
        Abstract method to skeletonize a given cell_id and return the corresponding mesh/skel
        """
        pass

    @abstractmethod
    def get_synapses(self, *args, **kwargs) -> pd.DataFrame:
        """
        Abstract method to get synapses for a given cell_id from
        a data source and return the full DataFrame.
        """
        pass

    @abstractmethod
    def get_neuron_by_id(self, *args, **kwargs) -> Neuron:
        """
        Abstract method to a neuron by cell_id
        """
        pass
