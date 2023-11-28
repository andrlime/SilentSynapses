import trimesh
import skeletor

import networkx as nx
from scipy.spatial.distance import euclidean


class Neuron:
    def __init__(self, mesh, skeleton, pre_synapse_locations, post_synapse_locations):
        """
        A single neuron object with mesh and skeleton information, and
        a table with the location of all proofread synapses.

        Parameters
        ----------
        mesh (CloudVolume Mesh):
            The mesh of the entire neuron
        skeleton (Skeletor Skeleton):
            The skeleton of the entire neuron
        pre_synapse_locations (pd.Dataframe):
            Metadata for pre synaptical sites
        post_synapse_locations (pd.Dataframe):
            Metadata for post synaptical sites
        """
        self.mesh = mesh
        self.skeleton = skeleton
        self.graph = self.convert_skeleton_to_nx_graph(skeleton)
        self.pre_synapses = pre_synapse_locations
        self.post_synapses = post_synapse_locations

    def convert_skeleton_to_nx_graph(self, skeleton):
        """
        Converts a Skeletor Skeleton object to a NetworkX weighted graph,
        where edge weights are distance between nodes.

        Parameters
        ----------
        skeleton (Skeletor.Skeleton):
            The skeleton representation of the Neuron

        Returns
        -------
        A graph G(V, E), where V = {id, x, y, z}, E = euclidean distance
            That is, the vertex is a dictionary with the x, y, and z coordinates of
            the node, as to make accessing it later easy.
        """
        swc_table = skeleton.swc
        graph = nx.Graph()

        for row in swc_table.iloc:
            graph.add_node(row.node_id, **row)

        for row in swc_table.iloc:
            if row.parent_id not in graph.nodes:
                continue

            own_location = [
                graph.nodes[row.node_id]["x"],
                graph.nodes[row.node_id]["y"],
                graph.nodes[row.node_id]["z"],
            ]
            parent_location = [
                graph.nodes[row.parent_id]["x"],
                graph.nodes[row.parent_id]["y"],
                graph.nodes[row.parent_id]["z"],
            ]

            distance_value = euclidean(own_location, parent_location)

            graph.add_edge(row.node_id, row.parent_id, weight=distance_value)
            graph.add_edge(row.parent_id, row.node_id, weight=distance_value)

        return graph
