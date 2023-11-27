import trimesh
import skeletor


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
        self.pre_synapses = pre_synapse_locations
        self.post_synapses = post_synapse_locations
