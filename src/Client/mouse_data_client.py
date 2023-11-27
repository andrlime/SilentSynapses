from abc import ABC, abstractmethod
import cloudvolume
from caveclient import CAVEclient

import pandas as pd
import numpy as np
import skeletor as sk
from skeletor import Skeleton
import os
import trimesh

from src.Client.data_client import DataClient
from src.Neuron.neuron import Neuron


def parse_swc_content(swc_content):
    """
    Given the content of an SWC file, create the corresponding SWC table.

    Parameters
    ----------
    swc_content : string
        Content of the swc representation of the skeleton as a string
    """
    lines = swc_content.split("\n")
    filtered_lines = [
        line for line in lines if not line.startswith("#") and line.strip()
    ]
    skeleton_data = np.stack(
        [np.array(line.split(), dtype=float) for line in filtered_lines]
    )

    columns = ["node_id", "label", "x", "y", "z", "radius", "parent_id"]
    skeleton_df = pd.DataFrame(skeleton_data, columns=columns)

    # Drop the 'label' column
    skeleton_df.drop("label", axis=1, inplace=True)

    return skeleton_df


class MouseDataClient(DataClient):
    def __init__(self, cv_uri, cave_uri, cave_api_key, cache_folder):
        """
        Initializes client to fetch mouse synapse data from a given CloudVolume source.

        Parameters
        ----------
        cv_uri (str):
            The URI for the CloudVolume data source.
        cave_uri (str):
            The URI for the CAVE data source.
        cave_api_key (str):
            The API key for authenticating with the CAVE source.
        cache_folder (str):
            The path to the folder used for caching data.
        """
        self.cv_uri = cv_uri
        self.cave_uri = cave_uri

        self.cv_connection = cloudvolume.CloudVolume(cv_uri)

        try:
            self.cave_connection = CAVEclient()
            self.cave_connection.auth.save_token(cave_api_key)
        except ValueError as e:
            print(e)

        self.cave_connection = CAVEclient(cave_uri)

        self.cache_folder = cache_folder

    def change_cloud_uri(self, new_uri) -> None:
        """
        Method for connecting to a new CloudVolume data source.
        
        Parameters
        ----------
        new_uri (str):
            The new CloudVolume URI
        """
        self.cv_connection = cloudvolume.CloudVolume(new_uri)

    def change_cave_uri(self, new_uri) -> None:
        """
        Method for connecting to a new CAVEclient data source.
        
        Parameters
        ----------
        new_uri (str):
            The new CAVEclient URI
        """
        self.cave_connection = CAVEclient(new_uri)

    def get_proofread_neurons(self):
        """
        Method to fetch and save proofread neurons from a given data source
        
        Parameters
        ----------
        (none)
        """
        proofreads = self.cave_connection.materialize.query_table(
            "proofreading_status_public_release",
            filter_equal_dict={"status_axon": "extended"},
        )

        self.proofread = proofreads
        return proofreads

    def skeletonize(self, cell_id, force_fresh=False):
        """
        Skeletonize a given cell_id. If already in cache, then restore cached one.
        Else, skeletonize it and cache as a file. Returns both the mesh and skeleton.

        Parameters
        ----------
        cell_id (int):
            The Neuron ID to skeletonize
        force_fresh (bool, default=False):
            Whether to forcibly not used cached files (in some cases, can be faster)
        """
        # Attempt to fetch mesh from cache. If mesh does
        # not exist in cache, assume skeleton is trash too.
        mesh_0_file_path = os.path.expanduser(
            f"{self.cache_folder}/meshes/{cell_id}_lod0.obj"
        )
        mesh_1_file_path = os.path.expanduser(
            f"{self.cache_folder}/meshes/{cell_id}_lod1.obj"
        )
        skel_file_path = os.path.expanduser(
            f"{self.cache_folder}/skeletons/{cell_id}.swc"
        )

        mesh_cached = False  # If a cached version of the mesh was found
        skel_cached = False  # If a cached version of the skeleton was found

        mesh = None
        mesh_lod_1 = None
        skel = None

        if (
            os.path.isfile(mesh_0_file_path)
            and os.path.isfile(mesh_1_file_path)
            and (not force_fresh)
        ):
            # Fetch meshes from cache
            print(f"Attempting to fetch mesh {cell_id} from cache")
            try:
                f = open(mesh_0_file_path, "r")
                mesh = cloudvolume.Mesh.from_obj(f.read())
                f.close()

                f = open(mesh_1_file_path, "r")
                mesh_1_from_obj = cloudvolume.Mesh.from_obj(f.read())
                mesh_lod_1 = mesh_1_from_obj
                f.close()

                print(f"Successfully fetched mesh {cell_id} from cache")
                mesh_cached = True
            except Error as e:
                print(f"Unexpected error when fetching mesh {cell_id} from cache")
                mesh_cached = False

            if os.path.isfile(skel_file_path) and mesh_cached is True:
                # Also fetch skeleton from path
                # If cached mesh is bad, the skeleton is likely bad too
                print(f"Attempting to fetch skeleton {cell_id} from cache")
                try:
                    f = open(skel_file_path, "r")
                    swc_from_file = f.read()
                    skel = Skeleton(
                        swc=parse_swc_content(swc_from_file),
                        mesh=mesh_lod_1,
                        method="wavefront",
                    )  # TODO: Fix
                    print(f"Successfully fetched skeleton {cell_id} from cache")
                    skel_cached = True
                except Error as e:
                    print(
                        f"Unexpected error when fetching skeleton {cell_id} from cache"
                    )
                    skel_cached = False

        if not mesh_cached:
            print(f"Fetching fresh mesh for cell_id = {cell_id}")
            mesh = self.cv_connection.mesh.get(cell_id, lod=0)[
                cell_id
            ]  # The actual mesh
            mesh_lod_1 = self.cv_connection.mesh.get(cell_id, lod=1)[
                cell_id
            ]  # Higher LOD for skeletonizing

            os.makedirs(os.path.dirname(mesh_0_file_path), exist_ok=True)

            mesh_0_obj = mesh.to_obj()
            mesh_1_obj = mesh_lod_1.to_obj()

            f = open(mesh_0_file_path, "wb")
            f.write(mesh_0_obj)
            f.close()

            f = open(mesh_1_file_path, "wb")
            f.write(mesh_1_obj)
            f.close()

        if not mesh_cached or not skel_cached:
            print(f"Generating new skeleton for cell_id = {cell_id}")
            if isinstance(mesh_lod_1, type(None)):
                raise Exception("Unexpected None type for mesh_lod_1")

            fixed = sk.pre.fix_mesh(mesh_lod_1, inplace=False)
            skel = sk.skeletonize.by_wavefront(
                fixed, waves=1, step_size=1
            )  # TODO: Investigate using more waves

            os.makedirs(os.path.dirname(skel_file_path), exist_ok=True)
            skel.save_swc(skel_file_path)

        if isinstance(mesh, type(None)):
            raise Exception("Unexpected None type for mesh")

        if isinstance(skel, type(None)):
            raise Exception("Unexpected None type for skel")

        return mesh, skel

    def get_synapses(self, cell_id, is_pre=False) -> pd.DataFrame:
        """
        Get synapses from a data source and return the full DataFrame.
        
        Parameters
        ----------
        cell_id (int):
            The Neuron ID to skeletonize
        is_pre (bool, default=False):
            Whether to get pre synaptical sites if true, or post if false
        """
        if cell_id < 0:
            raise Exception(f"cell_id must be positive, but got {cell_id}")

        position_name = "pre_pt_position" if is_pre else "post_pt_position"
        other_position_name = "post_pt_position" if is_pre else "pre_pt_position"

        synapses_dataframe = None
        if is_pre:
            synapses_dataframe = self.cave_connection.materialize.synapse_query(
                pre_ids=cell_id
            )
        else:
            synapses_dataframe = self.cave_connection.materialize.synapse_query(
                post_ids=cell_id
            )

        if isinstance(synapses_dataframe, type(None)):
            raise Exception(
                "Unexpected None for synapses DataFrame. There might be a problem with CAVEclient, or maybe your internet connection."
            )

        # Explode the columns
        synapses = pd.DataFrame(
            synapses_dataframe[position_name].to_list(), columns=["x", "y", "z"]
        )

        # Rescale
        synapses.x = synapses.x * 4
        synapses.y = synapses.y * 4
        synapses.z = synapses.z * 40

        # Explode the columns
        other_positions = pd.DataFrame(
            synapses_dataframe[other_position_name].to_list(),
            columns=["other_x", "other_y", "other_z"],
        )

        # Rescale
        other_positions.other_x = other_positions.other_x * 4
        other_positions.other_y = other_positions.other_y * 4
        other_positions.other_z = other_positions.other_z * 40

        # Merge the locations
        synapses = pd.concat([synapses, other_positions], axis=1)

        # Add and convert 'id', 'pre_pt_root_id', and 'post_pt_root_id' to numeric, coercing errors to NaN
        synapses["id"] = pd.to_numeric(synapses_dataframe["id"], errors="coerce")
        synapses["pre_pt_root_id"] = pd.to_numeric(
            synapses_dataframe["pre_pt_root_id"], errors="coerce"
        )
        synapses["post_pt_root_id"] = pd.to_numeric(
            synapses_dataframe["post_pt_root_id"], errors="coerce"
        )

        # Drop rows with NaN in any of these columns
        synapses = synapses.dropna(subset=["id", "pre_pt_root_id", "post_pt_root_id"])

        # Ensure the columns are of integer type
        synapses["id"] = synapses["id"].astype(int)
        synapses["pre_pt_root_id"] = synapses["pre_pt_root_id"].astype(int)
        synapses["post_pt_root_id"] = synapses["post_pt_root_id"].astype(int)

        # Reordering columns to put 'id' first
        synapses = synapses[
            [
                "id",
                "pre_pt_root_id",
                "post_pt_root_id",
                "x",
                "y",
                "z",
                "other_x",
                "other_y",
                "other_z",
            ]
        ]

        # TODO: Add caching into cache/synapse_locations
        # Low priority, as this is a fast operation, while skeletonizing is much slower.

        return synapses

    def get_neuron_by_id(self, cell_id) -> Neuron:
        """
        Method to initialize a neuron by cell_id, wrapping everything

        Parameters
        ----------
        cell_id (int):
            The Neuron ID to turn into an object
        """
        mesh, skeleton = self.skeletonize(cell_id)
        pre_synapses = self.get_synapses(cell_id, is_pre=True)
        post_synapses = self.get_synapses(cell_id, is_pre=False)
        return Neuron(mesh, skeleton, pre_synapses, post_synapses)
