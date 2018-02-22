import os
import matplotlib as mpl
from pyresample import geometry

import warnings
warnings.filterwarnings("ignore")

# The local path of the module (e.g. for retrieving resources)
local_src_path = os.path.split(os.path.abspath(__file__))[0]
local_path = os.sep.join(local_src_path.split(os.sep)[:-1])
LOCAL_PATH_RESOURCES = os.path.join(local_path, "resources")
REGION_DEF_FILENAME = r"simre_region_definition.yaml"
RECONCILED_GRID_SUB_FOLDERS = ['reconciled', 'grid']

# SIMRE Grid settings
DEFAULT_GRID_DEFINITION = "nh25kmEASE2.yaml"
TARGET_AREA_DEF = geometry.AreaDefinition(
    'nh25kmease2', 'EASE2 North (25km)', 'nh25kmease2',
    {'lat_0': '90.00', 'lat_ts': '70.00',
     'lon_0': '0.00', 'proj': 'laea',
     'ellps': 'WGS84', 'datum': 'WGS84'},
    432, 432,
    [-5400000.0, -5400000.0, 5400000.0, 5400000.0])

# Matplotlib default settings
mpl.rcParams['font.sans-serif'] = "arial"
for target in ["xtick.color", "ytick.color", "axes.edgecolor",
               "axes.labelcolor"]:
    mpl.rcParams[target] = "#4b4b4d"
