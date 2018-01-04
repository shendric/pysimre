import os
import sys
import textwrap
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
        except SystemExit:
            sys.exit("Forced Exit (raise SystemExit or sys.exit() evoked)")
        except Exception as err:
            self.log.error("Unkown error: %s" % str(err))
        finally:
            # Always close the file
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
        """
        Writes the global attributes to the file
        """
        # Write Global Attributes
        self.rootgrp.setncattr("title", "Regional Data for Sea Ice Mass Reconciliation Exercise (SIMRE)")
        self.rootgrp.setncattr("project", "Arctic+ Theme 2: Sea Ice Mass")

        summary = (
            'Collection of reconciled sea ice thickness estimates and statistical parameters '
            'for the Sea Ice Mass Reconciliation exercise (SIMRE) for region `%s` and observational '
            'period: `%s`')
        summary = summary % (self.ensbl.region_id, self.ensbl.period_id)
        summary = textwrap.dedent(summary).strip()
        print summary

        self.rootgrp.setncattr("summary", summary)
        self.rootgrp.setncattr("region_id", self.ensbl.region_id)
        self.rootgrp.setncattr("region_label", self.ensbl.region_label)
        self.rootgrp.setncattr("period_id", self.ensbl.period_id)
        self.rootgrp.setncattr("cdm_data_type", "Grid")
        self.rootgrp.setncattr("spatial_resolution", "25.0 km grid spacing")
        self.rootgrp.setncattr("geospatial_bounds_crs", "EPSG:6931")
        self.rootgrp.setncattr("creator_name", "Stefan Hendricks")
        self.rootgrp.setncattr("creator_email", "stefan.hendricks@awi.de")
        self.rootgrp.setncattr("contributor_name", "TBD")
        self.rootgrp.setncattr("contributor_role", "TBD")
    
    def populate_variables(self):
        """ Write variables with metadata """

        # Get shape of variables
        shape = np.shape(self.ensbl.longitude)
        dimdict = OrderedDict([("lat", shape[0]), ("lon", shape[1])])
        lons, lats = self.ensbl.longitude, self.ensbl.latitude

        # Write dimensions
        dims = dimdict.keys()
        for key in dims:
            self.rootgrp.createDimension(key, dimdict[key])

        # Write Variables
        dim = tuple(dims[0:len(lons.shape)])
        dtype_str = lons.dtype.str
        varlon = self.rootgrp.createVariable("longitude", dtype_str, dim, zlib=True)
        setattr(varlon, "long_name", "longitude of grid cell center")
        setattr(varlon, "standard_name", "longitude")
        setattr(varlon, "units", "degrees east")
        varlon[:] = lons

        varlat = self.rootgrp.createVariable("latitude", dtype_str, dim, zlib=True)
        setattr(varlat, "long_name", "latitude of grid cell center")
        setattr(varlat, "standard_name", "latitude")
        setattr(varlat, "units", "degrees north")
        varlat[:] = lats

        varmsit = self.rootgrp.createVariable("mean_thickness", dtype_str, dim, zlib=True)
        setattr(varmsit, "long_name", "Mean thickness from ensemble members")
        setattr(varmsit, "standard_name", "sea_ice_thickness")
        setattr(varmsit, "units", "m")
        varmsit[:] = self.ensbl.mean_thickness

        vargmsit = self.rootgrp.createVariable("gmean_thickness", dtype_str, dim, zlib=True)
        setattr(vargmsit, "long_name", "Geometric mean thickness from ensemble members")
        setattr(vargmsit, "standard_name", "sea_ice_thickness")
        setattr(vargmsit, "units", "m")
        vargmsit[:] = self.ensbl.gmean_thickness

        varmedsit = self.rootgrp.createVariable("median_thickness", dtype_str, dim, zlib=True)
        setattr(varmedsit, "long_name", "Median thickness from ensemble members")
        setattr(varmedsit, "standard_name", "sea_ice_thickness")
        setattr(varmedsit, "units", "m")
        varmedsit[:] = self.ensbl.median_thickness

        varminsit = self.rootgrp.createVariable("min_thickness", dtype_str, dim, zlib=True)
        setattr(varminsit, "long_name", "Minimum thickness of all ensemble members")
        setattr(varminsit, "standard_name", "sea_ice_thickness")
        setattr(varminsit, "units", "m")
        varminsit[:] = self.ensbl.min_thickness

        varmaxsit = self.rootgrp.createVariable("max_thickness", dtype_str, dim, zlib=True)
        setattr(varmaxsit, "long_name", "Maximum thickness of all ensemble members")
        setattr(varmaxsit, "standard_name", "sea_ice_thickness")
        setattr(varmaxsit, "units", "m")
        varmaxsit[:] = self.ensbl.max_thickness

        varsitsdev = self.rootgrp.createVariable("sdev_thickness", dtype_str, dim, zlib=True)
        setattr(varsitsdev, "long_name", "Thickness standard deviation from ensemble members")
        setattr(varsitsdev, "units", "m")
        varsitsdev[:] = self.ensbl.thickness_stdev

        npts = self.ensbl.n_points
        varnpts = self.rootgrp.createVariable("n_points", npts.dtype.str, dim, zlib=True)
        setattr(varnpts, "long_name", "Number of ensemble members")
        setattr(varnpts, "units", "1")
        varnpts[:] = npts

    @property
    def output_folder(self):
        return str(self._folder)

    @property
    def output_filepath(self):
        filename = "simre-l3-sit-reconciled-%s-%s.nc" % (self.ensbl.region_id, self.ensbl.period_id)
        return os.path.join(self.output_folder, filename)

    @property
    def ensbl(self):
        return self._ensbl
