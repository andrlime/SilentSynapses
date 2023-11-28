import trimesh
import skeletor

from src.Neuron.neuron import Neuron


class NeuronStore:
    def __init__(self, client):
        """
        A storage container for neuron object to allow for caching in RAM, making things even faster
        then caching to disk.
        """
        self.client = client
        self.neurons = {}

    def add_neuron(self, cell_id):
        """Adds a neuron to the bank by cell_id

        Parameters
        ----------
        client : MouseDataClient
        The client used to retrieve neurons

        cell_id : number
            The Neuron ID to get
        """
        new_neuron = self.client.get_neuron_by_id(cell_id)  # TODO: support pre and post
        self.neurons[cell_id] = new_neuron
        return new_neuron

    def get_neuron(self, cell_id, add_new=True):
        """Gets a neuron to the bank by cell_id

        Parameters
        ----------
        client : MouseDataClient
            The client used to retrieve neurons

        cell_id : number
            The Neuron ID to get

        add_new : boolean (default: True)
            Whether to add the neuron if it does not exist
        """

        # Check if the neuron exists in the NeuronBank's 'neurons' dictionary
        if cell_id in self.neurons:
            # Neuron exists, retrieve and return it from the dictionary
            return self.neurons[cell_id]
        elif add_new:
            # Neuron does not exist, and add_new is True, add the neuron
            return self.add_neuron(
                cell_id
            )  # Assuming add_neuron is a method of this class

        # If the neuron does not exist and add_new is False, raise an exception
        raise Exception(
            f"Neuron with ID {cell_id} does not exist in bank, and option `add_new` is set to False."
        )

    def delete_neuron(self, cell_id):
        """Removes a neuron from the bank by cell_id

        Parameters
        ----------
        cell_id : number
            The Neuron ID to remove
        """

        # Check if the neuron exists in the dictionary
        if cell_id in self.neurons:
            # Remove the neuron
            del self.neurons[cell_id]
