import trimesh
import skeletor


class Neuron:
    def __init__(self, mesh, skeleton, pre_synapse_locations, post_synapse_locations):
        """
        A single neuron object with mesh and skeleton information, and
        a table with the location of all proofread synapses.
        """
        self.mesh = mesh
        self.skeleton = skeleton
        self.pre_synapses = pre_synapse_locations
        self.post_synapses = post_synapse_locations
