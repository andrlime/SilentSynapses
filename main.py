# Standard libraries
import itertools
import time
from collections import deque

# Third-party libraries
import cloudvolume
import matplotlib.pyplot as plt
import networkx as nx
import nglui
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import skeletor as sk
import trimesh
from caveclient import CAVEclient
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.spatial.distance import cdist, euclidean
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

# Timing
from tqdm import tqdm

# Environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# Local imports
from src.Client.mouse_data_client import MouseDataClient
from src.Processor.mouse_data_processor import MouseDataProcessor

# Set random state for numpy
rng = np.random.RandomState(1)

print("Successfully imported all libraries!")

API_KEY = os.getenv("CC_API")
CV_URI = (
    "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg"
)
CAVE_URI = "minnie65_public_v117"
CACHE = "~/silent_synapses/cache"
OUTPUT_FOLDER = "~/silent_synapses/out"
NUMBER_THREADS = 4

data_client = MouseDataClient(CV_URI, CAVE_URI, API_KEY, CACHE)
proofread_synapses = data_client.get_proofread_neurons()
data_processor = MouseDataProcessor(
    radius_of_interest=2500,
    pca_threshold=0.9,
    data_client=data_client,
    check_remote=False,
)


def main_function(cell_id):
    print(f"Running on cell {cell_id}")
    (
        self_pre_ratios,
        self_post_ratios,
        remote_pre_ratios,
        remote_post_ratios,
    ) = data_processor.measure_all_synapses(
        cell_id=cell_id
    )  # TODO: Multithread

    # Write to disk
    output_file_path = os.path.expanduser(f"{OUTPUT_FOLDER}/ratios/{cell_id}.csv")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    output_str = f"self_pre,self_post,remote_pre,remote_post\n"
    max_length = np.max(
        [
            len(self_pre_ratios),
            len(remote_pre_ratios),
            len(self_post_ratios),
            len(remote_post_ratios),
        ]
    )
    output_data = np.array(
        [
            np.pad(self_pre_ratios, (0, max_length - len(self_pre_ratios))),
            np.pad(remote_pre_ratios, (0, max_length - len(remote_pre_ratios))),
            np.pad(self_post_ratios, (0, max_length - len(self_post_ratios))),
            np.pad(remote_post_ratios, (0, max_length - len(remote_post_ratios))),
        ]
    )

    for output_row in output_data.T:
        output_str += (
            f"{output_row[0]},{output_row[1]},{output_row[2]},{output_row[3]}\n"
        )

    f = open(output_file_path, "w")
    f.write(output_str)
    f.close()

k = int(os.getenv("SLURM_ARRAY_TASK_ID"))
rows_per_fraction = (len(proofread_synapses) // NUMBER_THREADS)
start_idx = rows_per_fraction * (k - 1)
end_idx = start_idx + rows_per_fraction
print("Starting main loop...")
proofread_synapses.iloc[start_idx:end_idx].apply(lambda row: main_function(row["valid_id"]), axis=1)
