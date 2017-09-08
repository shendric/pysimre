import os

# The local path of the module (e.g. for retrieving resources)
local_src_path = os.path.split(os.path.abspath(__file__))[0]
local_path = os.sep.join(local_src_path.split(os.sep)[:-1])
