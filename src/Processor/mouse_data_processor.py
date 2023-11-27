from abc import ABC, abstractmethod

import pandas as pd
import networkx as nx
from tqdm import tqdm

from src.Processor.data_processor import DataProcessor
from src.Neuron.neuron_store import NeuronStore


class MouseDataProcessor(DataProcessor):
    def __init__(self, radius_of_interest, pca_threshold, data_client):
        """
        Initializes a processor for mouse neuron data. Configs the radius of interest.

        Parameters
        ----------
        radius_of_interest (float)
            The radius of interest for to extract synapses.
        pca_threshold (float)
            The threshold value for Principal Component Analysis.
        data_client (object)
            The data client used for accessing neuron data.
        """
        self.neuron_storage = NeuronStore(data_client)
        self.pca_threshold = pca_threshold
        self.radius_of_interest = radius_of_interest

    def extract_synapse(self, cell_id, synapse_location):
        """
        Method to extract a specific synapse given location and return
        the synapse as a NetworkX graph, along with the corresponding
        surrounding mesh.

        Parameters
        ----------
        cell_id (float):
            The neuron on which the synapse lies
        synapse_location ((float, float, float)):
            Coordinate of the synapse being extracted.
        """
        return False, False

    def filter_synapse(self, synapse_graph, synapse_mesh) -> bool:
        """
        Method to take a measured synapses and determine if it is valid.

        Filters out the "bad" data:
            1. Must be relatively straight, use PCA
            2. No acute angles, use dot product

        Parameters
        ----------
        synapse_graph (NetworkX Graph):
            Network representation of the synapse
        synapse_mesh (CloudVolume Mesh):
            Mesh representation of the surrounding region
        """
        pass

    def measure_head_to_neck_ratio(self, synapse_graph, synapse_mesh) -> float:
        """
        Method to, given a synapse and mesh, measure its head to neck ratio.

        Parameters
        ----------
        synapse_graph (NetworkX Graph):
            Network representation of the synapse
        synapse_mesh (CloudVolume Mesh):
            Mesh representation of the surrounding region
        """
        pass

    def measure_all_synapses(self, cell_id):
        """
        Method to measure all synapses head/neck ratios from a neuron.

        Parameters
        ----------
        cell_id (float):
            Neuron ID to measure all synapses for
        """
        neuron = self.neuron_storage.get_neuron(cell_id=cell_id)
        self_pre_ratios, self_post_ratios = [], []
        remote_pre_ratios, remote_post_ratios = [], []  # TODO

        print("Extracting pre synaptical sites")
        for synapse in tqdm(neuron.pre_synapses.iloc):
            extracted_graph, extracted_mesh = self.extract_synapse(
                cell_id, (synapse.x, synapse.y, synapse.z)
            )

            other_extracted_graph, other_extracted_mesh = self.extract_synapse(
                synapse.post_pt_root_id,
                (synapse.other_x, synapse.other_y, synapse.other_z),
            )

            if self.filter_synapse(extracted_graph, extracted_mesh):
                self_pre_ratios.append(
                    self.measure_head_to_neck_ratio(extracted_graph, extracted_mesh)
                )

            if self.filter_synapse(other_extracted_graph, other_extracted_mesh):
                remote_post_ratios.append(
                    self.measure_head_to_neck_ratio(
                        other_extracted_graph, other_extracted_mesh
                    )
                )

        print("Extracting post synaptical sites")
        for synapse in tqdm(neuron.post_synapses.iloc):
            extracted_graph, extracted_mesh = self.extract_synapse(
                cell_id, (synapse.x, synapse.y, synapse.z)
            )

            other_extracted_graph, other_extracted_mesh = self.extract_synapse(
                synapse.pre_pt_root_id,
                (synapse.other_x, synapse.other_y, synapse.other_z),
            )

            if self.filter_synapse(extracted_graph, extracted_mesh):
                self_post_ratios.append(
                    self.measure_head_to_neck_ratio(extracted_graph, extracted_mesh)
                )

            if self.filter_synapse(other_extracted_graph, other_extracted_mesh):
                remote_pre_ratios.append(
                    self.measure_head_to_neck_ratio(
                        other_extracted_graph, other_extracted_mesh
                    )
                )

        return self_pre_ratios, self_post_ratios, remote_pre_ratios, remote_post_ratios
