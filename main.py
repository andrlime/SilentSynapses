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
CV_URI = "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/seg"
CAVE_URI = "minnie65_public_v117"
CACHE = "~/silent_synapses/cache"

data_client = MouseDataClient(CV_URI, CAVE_URI, API_KEY, CACHE)
proofread_synapses = data_client.get_proofread_neurons()
data_processor = MouseDataProcessor(radius_of_interest=2500, pca_threshold=0.9, data_client=data_client)

print(data_processor.measure_all_synapses(cell_id=864691136194248918))
