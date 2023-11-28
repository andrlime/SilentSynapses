from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import networkx as nx
import trimesh
from tqdm import tqdm
from scipy.spatial.distance import cdist, euclidean
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from src.Processor.data_processor import DataProcessor
from src.Neuron.neuron_store import NeuronStore
from matplotlib import pyplot as plt


def draw_and_save_graph(subgraph, node_id, highlights):
    # Draw the graph
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    pos = nx.spring_layout(subgraph)  # Positions for all nodes

    # Draw all nodes. Nodes not in highlights will be in a default color (e.g., blue)
    nx.draw_networkx_nodes(subgraph, pos, node_color="blue")

    # Highlight specified nodes in red
    if highlights:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=highlights, node_color="red")

    # Draw edges and labels
    nx.draw_networkx_edges(subgraph, pos)
    nx.draw_networkx_labels(subgraph, pos)

    # Set title and labels (optional)
    plt.title(f"Subgraph around Node {node_id}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Save the figure
    plt.savefig(f"{node_id}.png")


def pca_cloud(filtered_coords, pca_threshold=0.9):
    """
    Does PCA analysis on the coordinates and returns true if the
    max variance > pca_threshold

    Parameters
    ----------
    filtered_coordinates:
        [x, y, z] coordinates
    pca_threshold:
        value in [0, 1] for PCA, default is 0.9
    """
    # First, we need the skeleton's coordinates to be formatted
    coordinates = np.column_stack(
        (filtered_coords.x, filtered_coords.y, filtered_coords.z)
    )

    # Now we perform PCA
    pca = PCA(n_components=3)
    pca.fit(coordinates)

    # Calculate the normalized explained variance
    explained_variance_ratio = pca.explained_variance_ratio_

    # Find the maximum absolute value in the explained variance ratio
    max_var = np.max(np.abs(explained_variance_ratio))

    return max_var > pca_threshold


def angle_between_vectors(v1, v2):
    """
    Pretty obvious what this does
    """
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return angle_rad * 180 / np.pi


def contains_acute_angles(graph):
    """
    Returns true if the graph contains any acute angles

    Parameters
    ----------
    graph (nx.Graph):
        A graph with node_id as key and {x, y, z} as data
    """
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 2:
            vector1 = np.array(
                [
                    graph.nodes[neighbors[0]]["x"],
                    graph.nodes[neighbors[0]]["y"],
                    graph.nodes[neighbors[0]]["z"],
                ]
            ) - np.array(
                [graph.nodes[node]["x"], graph.nodes[node]["y"], graph.nodes[node]["z"]]
            )
            vector2 = np.array(
                [
                    graph.nodes[neighbors[1]]["x"],
                    graph.nodes[neighbors[1]]["y"],
                    graph.nodes[neighbors[1]]["z"],
                ]
            ) - np.array(
                [graph.nodes[node]["x"], graph.nodes[node]["y"], graph.nodes[node]["z"]]
            )
            angle = angle_between_vectors(vector1, vector2)
            if angle < 90:
                return True
    return False


class MouseDataProcessor(DataProcessor):
    def __init__(self, radius_of_interest, pca_threshold, data_client, check_remote):
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
        check_remote (bool)
            Whether to check the other end for each synapse
        """
        self.radius_of_interest = radius_of_interest
        self.pca_threshold = pca_threshold
        self.neuron_storage = NeuronStore(data_client)
        self.check_remote = check_remote

    def longest_path(self, original_graph, closest_node):
        """
        Finds the longest path from a node to the dendritic spine

        Parameters
        ----------
        original_graph (nx.Graph):
            The neuron
        closest_node (int):
            closest_node ID
        """
        # The center-point location, around which we only look so far
        ref_x, ref_y, ref_z = (
            original_graph.nodes[closest_node]["x"],
            original_graph.nodes[closest_node]["y"],
            original_graph.nodes[closest_node]["z"],
        )
        reference_point = (ref_x, ref_y, ref_z)

        # Find nodes within the radius using list comprehension
        nodes_within_radius = [
            n
            for n, attrs in original_graph.nodes(data=True)
            if euclidean(reference_point, (attrs["x"], attrs["y"], attrs["z"]))
            <= self.radius_of_interest
        ]

        # Create subgraph
        G = original_graph.subgraph(nodes_within_radius)
        visited_nodes = set()

        def dfs(node, path):
            nonlocal G, visited_nodes
            visited_nodes.add(node)
            neighbors = [
                n
                for n in G.neighbors(node)
                if n not in visited_nodes and (len(path) < 2 or n != path[-2])
            ]
            extended_paths = [dfs(n, path + [n]) for n in neighbors]

            if len(list(G.neighbors(node))) > 3:
                return path + [node]

            if not extended_paths:
                return path

            longest_path = max(extended_paths, key=len)
            return longest_path

        # Perform DFS traversal in both directions from the closest_node
        main_spines = []
        for neighbor in G.neighbors(closest_node):
            visited_nodes = set([closest_node])
            path = dfs(neighbor, [closest_node, neighbor])
            main_spines.append(path)

        # Combine the paths from both directions
        main_spine = main_spines[0][:-1] + main_spines[1]

        # TODO: Add some drawing config and better plots
        # draw_and_save_graph(G, closest_node, main_spine)

        return main_spine

    def prune_leaves_until_branch(self, G, main_spine):
        """
        Prunes leaves from the main_spine

        Parameters
        ----------
        G (nx.Graph):
            The neuron
        main_spine Array<int>:
            The synapse spine
        """
        pruned_spine = main_spine.copy()

        start_node = pruned_spine[0]
        while True:
            end_node = pruned_spine[-1]

            degree = G.degree(end_node)

            # If the degree is greater than 2, we've reached a branch
            if degree > 2:
                break

            if end_node == start_node:
                return main_spine

            # Remove the leaf node from the spine
            pruned_spine.pop()

        return pruned_spine

    def extract_synapse(self, cell_id, synapse_location):
        """
        Method to extract a specific synapse given location and return
        the synapse as a NetworkX graph, along with the corresponding
        surrounding mesh and sub dataframe for the new graph.

        Parameters
        ----------
        cell_id (float):
            The neuron on which the synapse lies
        synapse_location ((float, float, float)):
            Coordinate of the synapse being extracted.
        """
        # Convert the skeleton to NetworkX WGraph
        neuron = self.neuron_storage.get_neuron(cell_id=cell_id)
        neuron_graph = neuron.graph.copy()
        mesh = neuron.mesh

        # Find the closest Node
        skeleton_df = neuron.skeleton.swc
        distances = skeleton_df.apply(
            lambda row: euclidean(synapse_location, [row["x"], row["y"], row["z"]]),
            axis=1,
        )
        closest_node = int(skeleton_df.loc[distances.idxmin()]["node_id"])

        # Find the main spine as the longest path between a leaf node and a branch node TODO: fix
        main_spine = self.longest_path(neuron_graph, closest_node)
        main_spine = self.prune_leaves_until_branch(neuron_graph, main_spine)

        # Prune nodes not in the main spine
        nodes_to_remove = [
            node
            for node in neuron_graph.nodes
            if int(node) not in np.array(main_spine).astype(int)
        ]

        neuron_graph.remove_nodes_from(nodes_to_remove)
        the_synapse_itself = neuron_graph

        # Get the nearby mesh... this part is verbose
        def get_nearby_mesh_vertices(mesh, coords, radius):
            """
            Find and return mesh vertices that are within a specified radius of given coordinates.

            Parameters
            ----------
            mesh (CloudVolume Mesh):
                A mesh object with a 'vertices' attribute, which is a numpy array of vertex coordinates.
            coords (np.ndarray):
                A numpy array of coordinates (points) to calculate distances from.
            radius (float):
                The radius within which vertices will be considered 'nearby'.
            """
            # Calculate the distance from each mesh vertex to the provided coordinates
            distances = cdist(coords, mesh.vertices)

            # Determine which vertices are within the specified radius
            within_radius = np.any(distances < radius, axis=0)

            # Extract the vertices that are within the radius
            nearby_vertices = mesh.vertices[within_radius]

            # Extract the indices of these vertices
            nearby_indices = np.where(within_radius)[0]

            return nearby_vertices, nearby_indices

        # Extract the part of the mesh by filtering some coordinates
        smaller_rad = self.radius_of_interest / 5  # TODO: add as class member
        filtered_coordinates = skeleton_df[
            skeleton_df["node_id"].isin(np.array(main_spine).astype(int))
        ]
        coordinates_cleaned = filtered_coordinates[["x", "y", "z"]].to_numpy()

        nearby_mesh, nearby_indices = get_nearby_mesh_vertices(
            mesh, coordinates_cleaned, smaller_rad
        )
        face_inds = (
            np.isin(mesh.faces[:, 0], nearby_indices)
            * np.isin(mesh.faces[:, 1], nearby_indices)
            * np.isin(mesh.faces[:, 2], nearby_indices)
        )
        faces = mesh.faces[face_inds]

        def remap_ids(faces, where_inds):
            """
            Remap the vertex indices in mesh faces to a new set of indices.

            Parameters
            ----------
            faces (np.ndarray):
                A 2D numpy array where each row represents a face, and
                each column in a row represents a vertex index of that face.
            where_inds (np.ndarray):
                An array of original vertex indices which will be mapped to
                new indices starting from 0.
            """
            # Determine the new indices and the size of the faces array
            n_inds = np.shape(where_inds)[0]
            n_faces, n_dims = np.shape(faces)
            new_ids = np.arange(n_inds)

            # Create a dictionary to map old indices to new indices
            replacer = dict(zip(np.array(where_inds), new_ids))

            # Initialize an array for the new faces
            new_faces = np.zeros(np.shape(faces), int)

            # Replace old indices with new indices in each dimension of the faces
            for i in range(n_dims):
                new_faces[:, i] = np.array(
                    list(map(replacer.get, faces[:, i], faces[:, i]))
                )

            return new_faces

        # Fix up the mesh by changing face IDs
        nearby_faces = remap_ids(faces, nearby_indices)
        the_mesh_itself = (nearby_mesh, nearby_faces)

        return the_synapse_itself, the_mesh_itself, filtered_coordinates

    def filter_synapse(self, filtered_coordinates, synapse_graph) -> bool:
        """
        Method to take a measured synapse and determine if it is valid.

        Filters out the "bad" data:
            1. Must be relatively straight, use PCA
            2. No acute angles, use dot product

        Parameters
        ----------
        filtered_coordinates:
            Coordinates to do PCA on
        synapse_graph (NetworkX Graph):
            Network representation of the synapse
        """
        if isinstance(synapse_graph, type(None)):
            raise Exception("Unexpected NoneType for synapse_graph")

        if isinstance(filtered_coordinates, type(None)):
            raise Exception("Unexpected NoneType for filtered_coordinates")

        if len(synapse_graph.nodes) <= 5:
            print("Graph too short, rejecting")
            return False

        pca_ok = pca_cloud(filtered_coordinates, self.pca_threshold)
        angles_bad = contains_acute_angles(synapse_graph)
        return pca_ok and (not angles_bad)

    def measure_head_to_neck_ratio(
        self, center_point, synapse_graph, synapse_mesh
    ) -> float:
        """
        Method to, given a synapse and mesh, measure its head to neck ratio.

        Parameters
        ----------
        center_point ([float, float, float]):
            The approximate location of the synapse as determined
            by the proofread data. It is assumed, perhaps incorrectly
            so, that the center_point is closer to the head than the tail.
        synapse_graph (NetworkX Graph):
            Network representation of the synapse
        synapse_mesh ((mesh, faces)):
            Mesh representation of the surrounding region - returned from
            previous function as a tuple (mesh, faces)
        """
        ending_nodes = [node for node, degree in synapse_graph.degree() if degree == 1]
        farthest_node = 0
        farthest_dist = 0

        for node in ending_nodes:
            pos = np.array(
                (
                    synapse_graph.nodes[node]["x"],
                    synapse_graph.nodes[node]["y"],
                    synapse_graph.nodes[node]["z"],
                )
            )
            cntr = np.array(list(center_point))
            difference_in_psn = pos - cntr
            difference_squared = (
                (difference_in_psn[0] ** 2)
                + (difference_in_psn[1] ** 2)
                + (difference_in_psn[2] ** 2)
            )
            if (np.sqrt(difference_squared)) > farthest_dist:
                farthest_node = node

        def traverse_linear_graph(G, start_node):
            """
            Traverse the graph in order. Suppose A is the farthest node
            in a graph with [A - B - C - D - E], then this just returns
            [A, B, C, D, E]

            Parameters
            ----------
            G (nx.Graph):
                The graph
            start_node (node_id / int):
                The node to start from
            """
            # Initialize the traversal
            visited_nodes = [start_node]
            current_node = start_node

            # Traverse the graph
            while len(visited_nodes) < G.number_of_nodes():
                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited_nodes:
                        visited_nodes.append(neighbor)
                        current_node = neighbor
                        break

            return visited_nodes

        nodes_in_order = traverse_linear_graph(synapse_graph, farthest_node)

        counter = 0
        all_areas = []
        for node_index in range(len(nodes_in_order) - 1):
            # Get normal vector
            node1 = nodes_in_order[node_index]
            node2 = nodes_in_order[node_index + 1]
            normal_vector = [
                synapse_graph.nodes[node2]["x"] - synapse_graph.nodes[node1]["x"],
                synapse_graph.nodes[node2]["y"] - synapse_graph.nodes[node1]["y"],
                synapse_graph.nodes[node2]["z"] - synapse_graph.nodes[node1]["z"],
            ]

            # Normalize the vector by dividing each component by the magnitude
            magnitude = np.linalg.norm(normal_vector)
            normalized_vector = normal_vector / magnitude

            # Create a rotation matrix to make it flat
            dX, dY, dZ = (
                normalized_vector[0],
                normalized_vector[1],
                normalized_vector[2],
            )
            angle_x = np.arccos(dZ / (dY * dY + dZ * dZ) ** 0.5)
            angle_y = np.arccos(dZ / (dX * dX + dZ * dZ) ** 0.5)
            angle_z = np.arccos(dX / (dY * dY + dX * dX) ** 0.5)
            rotation_matrix = R.from_euler("xyz", [angle_x, angle_y, angle_z])

            iteration_count = 5  # TODO: maybe add to class, but doesn't really matter
            for j in range(iteration_count):
                cntr_pnt = [
                    synapse_graph.nodes[node1]["x"]
                    + (j / iteration_count) * normal_vector[0],
                    synapse_graph.nodes[node1]["y"]
                    + (j / iteration_count) * normal_vector[1],
                    synapse_graph.nodes[node1]["z"]
                    + (j / iteration_count) * normal_vector[2],
                ]

                mesh_trimesh = trimesh.Trimesh(
                    vertices=synapse_mesh[0], faces=synapse_mesh[1]
                )
                slice_trimesh = mesh_trimesh.section(
                    plane_origin=cntr_pnt, plane_normal=([dX, dY, dZ])
                )

                if slice_trimesh is None:
                    continue
                else:
                    rotated_xyz = []
                    for k in np.array(slice_trimesh.vertices):
                        rotated_xyz.append(rotation_matrix.apply(k))

                    rotated_xyz = np.transpose(rotated_xyz)
                    hull = ConvexHull(np.transpose([rotated_xyz[0], rotated_xyz[1]]))
                    all_areas.append([counter, hull.area**0.5])
                    counter += 1

        def compute_min_max(l):
            """
            Compute the min, max, and ratio

            Parameters
            ----------
            l:
                The list
            """
            if l[0] > l[len(l) - 1]:
                l = np.flip(l)

            min_val, min_idx = min((value, idx) for idx, value in enumerate(l))
            max_val, max_idx = max(
                (value, idx) for idx, value in enumerate(l[min_idx:])
            )

            return min_val, max_val, max_val / min_val

        moving_avg = []
        window_size = 3
        raw_data = np.array(all_areas).T[1]
        for j in range(len(raw_data) - 4):
            moving_avg.append(np.sum(raw_data[j : j + 4]) / 4)

        min_width, max_width, ratio = compute_min_max(moving_avg)
        return ratio

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
        remote_pre_ratios, remote_post_ratios = [], []

        fetch_attempt, fetch_fail = 0, 0

        print("Extracting pre synaptical sites")
        for synapse in tqdm(neuron.pre_synapses.iloc):
            fetch_attempt += 1

            try:
                syn_location = [synapse.x, synapse.y, synapse.z]
                (
                    extracted_graph,
                    extracted_mesh,
                    extracted_coords,
                ) = self.extract_synapse(cell_id, syn_location)

                if self.filter_synapse(extracted_coords, extracted_graph):
                    print(f"Accepted synapse {synapse.id}")
                    measurement = self.measure_head_to_neck_ratio(
                        syn_location, extracted_graph, extracted_mesh
                    )
                    print(f"Found ratio {measurement}")
                    self_pre_ratios.append(measurement)
                else:
                    print(f"Rejected synapse {synapse.id}")
            except Exception as e:
                print(f"error for synapse {synapse.id} - {e}")
                continue

            if self.check_remote:
                try:
                    syn_location = [synapse.other_x, synapse.other_y, synapse.other_z]
                    (
                        other_extracted_graph,
                        other_extracted_mesh,
                        other_extracted_coords,
                    ) = self.extract_synapse(
                        synapse.post_pt_root_id,
                        syn_location,
                    )

                    if self.filter_synapse(
                        other_extracted_coords, other_extracted_graph
                    ):
                        print(f"Accepted synapse {synapse.id}")
                        measurement = self.measure_head_to_neck_ratio(
                            syn_location, other_extracted_graph, other_extracted_mesh
                        )
                        print(f"Found ratio {measurement}")
                        remote_post_ratios.append(measurement)
                    else:
                        print(f"Rejected synapse {synapse.id}")
                except Exception as e:
                    fetch_fail += 1
                    print(f"error: {e}")
                    continue

        print("Extracting post synaptical sites")
        for synapse in tqdm(neuron.post_synapses.iloc):
            fetch_attempt += 1

            try:
                syn_location = [synapse.x, synapse.y, synapse.z]
                (
                    extracted_graph,
                    extracted_mesh,
                    extracted_coords,
                ) = self.extract_synapse(cell_id, syn_location)

                if self.filter_synapse(extracted_coords, extracted_graph):
                    print(f"Accepted synapse {synapse.id}")
                    measurement = self.measure_head_to_neck_ratio(
                        syn_location, extracted_graph, extracted_mesh
                    )
                    print(f"Found ratio {measurement}")
                    remote_post_ratios.append(measurement)
                else:
                    print(f"Rejected synapse {synapse.id}")
            except Exception as e:
                print(f"error for synapse {synapse.id} - {e}")
                continue

            if self.check_remote:
                try:
                    syn_location = [synapse.other_x, synapse.other_y, synapse.other_z]
                    (
                        other_extracted_graph,
                        other_extracted_mesh,
                        other_extracted_coords,
                    ) = self.extract_synapse(
                        synapse.pre_pt_root_id,
                        syn_location,
                    )

                    if self.filter_synapse(
                        other_extracted_coords, other_extracted_graph
                    ):
                        print(f"Accepted synapse {synapse.id}")
                        measurement = self.measure_head_to_neck_ratio(
                            syn_location, other_extracted_graph, other_extracted_mesh
                        )
                        remote_pre_ratios.append(measurement)
                    else:
                        print(f"Rejected synapse {synapse.id}")
                except Exception as e:
                    fetch_fail += 1
                    print(f"error: {e}")
                    continue

        print(
            f"Failure rate in fetching non origin synapses - {fetch_fail/fetch_attempt}"
        )

        return self_pre_ratios, self_post_ratios, remote_pre_ratios, remote_post_ratios
