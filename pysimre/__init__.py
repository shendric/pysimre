import os
import matplotlib as mpl

# The local path of the module (e.g. for retrieving resources)
local_src_path = os.path.split(os.path.abspath(__file__))[0]
local_path = os.sep.join(local_src_path.split(os.sep)[:-1])
LOCAL_PATH_RESOURCES = os.path.join(local_path, "resources")
REGION_DEF_FILENAME = r"simre_region_definition.yaml"
RECONCILED_GRID_SUB_FOLDERS = ['reconciled', 'grid']

# Matplotlib default settings
mpl.rcParams['font.sans-serif'] = "arial"
for target in ["xtick.color", "ytick.color", "axes.edgecolor",
               "axes.labelcolor"]:
    mpl.rcParams[target] = "#4b4b4d"
