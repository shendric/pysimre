import os
import sys
from netCDF4 import Dataset, num2date
from collections import defaultdict, OrderedDict
import numpy as np

from pysimre.misc import ClassTemplate


class GridReconciledNetCDF(ClassTemplate):

    def __init__(self, ensemble):
        super(GridReconciledNetCDF, self).__init__(self.__class__.__name__)
        self._ensbl = ensemble
        self._folder = None
        self.rootgrp = None

    def write(self, folder):
        """
        Writes the reconciled netCDF to specified folder. 
            :param folder: Local path (best use `reconciled_grid_dir` property from repo object) 
        """
        
        # safe parameter
        self._folder = folder
        
        # Open the file
        self.log.info("Write SIMRE reconciled netcdf: %s" % self.output_filepath)        
        self.rootgrp = Dataset(self.output_filepath, "w")
        try:
            self.populate_global_attributes()
            self.populate_variables()
        except:
            error = sys.exc_info()[0]
            self.log.error("Unkown error: %s" % str(error))
        finally:
            # Close the file
            self.rootgrp.close()

        ''' # Get metadata
        shape = np.shape(self.longitude)
        dimdict = OrderedDict([("lat", shape[0]), ("lon", shape[1])])
        if self.source_filename is list:
            source_filename = ", ".join(self.source_filename)
        else:
            source_filename = self.source_filename

        lons, lats = self.longitude, self.latitude

        self.log.info("Write SIMRE netcdf: %s" % self.output_filepath
        # Open the file
        rootgrp = Dataset(self.output_filepath, "w")

        # Write Global Attributes
        rootgrp.setncattr("title", "Regional Data for Sea Ice Mass " +
                          "Reconciliation Exercise (SIMRE)")
        rootgrp.setncattr("project", "Arctic+ Theme 2: Sea Ice Mass")
        rootgrp.setncattr("source", source_filename)
        rootgrp.setncattr("dataset_id", self.dataset_id)
        rootgrp.setncattr("region_id", self.region_id)
        rootgrp.setncattr("period_id", self.period_id)
        rootgrp.setncattr("summary", "TBD")
        rootgrp.setncattr("creator_name", "TBD")
        rootgrp.setncattr("creator_url", "TBD")
        rootgrp.setncattr("creator_email", "TBD")
        rootgrp.setncattr("contributor_name", "TBD")
        rootgrp.setncattr("contributor_role", "TBD")

        # Write dimensions
        dims = dimdict.keys()
        for key in dims:
            rootgrp.createDimension(key, dimdict[key])

        # Write Variables
        dim = tuple(dims[0:len(lons.shape)])
        dtype_str = lons.dtype.str
        varlon = rootgrp.createVariable("longitude", dtype_str, dim, zlib=True)
        setattr(varlon, "long_name", "longitude of grid cell center")
        setattr(varlon, "standard_name", "longitude")
        setattr(varlon, "units", "degrees")
        setattr(varlon, "scale_factor", 1.0)
        setattr(varlon, "add_offset", 0.0)
        varlon[:] = lons

        varlat = rootgrp.createVariable("latitude", dtype_str, dim, zlib=True)
        setattr(varlat, "long_name", "latitude of grid cell center")
        setattr(varlat, "standard_name", "latitude")
        setattr(varlat, "units", "degrees")
        setattr(varlat, "scale_factor", 1.0)
        setattr(varlat, "add_offset", 0.0)
        varlat[:] = lats

        varsit = rootgrp.createVariable("sea_ice_thickness",
                                        dtype_str, dim, zlib=True)
        setattr(varsit, "long_name", "thickness of the sea ice layer")
        setattr(varsit, "standard_name",  "sea_ice_thickness")
        setattr(varsit, "units", "meters")
        setattr(varsit, "scale_factor", 1.0)
        setattr(varsit, "add_offset", 0.0)
        varsit[:] = self.thickness

        # Close the file
        rootgrp.close() '''

    def populate_global_attributes(self):
        pass
    
    def populate_variables(self):
        pass

    @property
    def output_folder(self):
        return str(self._folder)

    @property
    def output_filepath(self):
        filename = "simre-l3-sit-reconciled-%s-%s.nc" % (self.enslb.region_id, self.enslb.period_id)
        return os.path.join(self.output_folder, filename)

    @property
    def enslb(self):
        return self._ensbl
