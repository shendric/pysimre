import os
import sys
import textwrap
from netCDF4 import Dataset, num2date, date2num
from collections import defaultdict, OrderedDict
import numpy as np

from pysimre.misc import ClassTemplate


class OrbitReconciledNetCDF(ClassTemplate):

    units = "seconds since 1970-01-01"
    calendar = "standard"

    def __init__(self, ensemble, calval_ensemble):
        super(OrbitReconciledNetCDF, self).__init__(self.__class__.__name__)
        self._ensbl = ensemble
        self._cv_ensbl = calval_ensemble
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
        self.populate_global_attributes()
        try:
            # self.populate_global_attributes()
            self.populate_variables()
        except SystemExit:
            sys.exit("Forced Exit (raise SystemExit or sys.exit() evoked)")
        except Exception as err:
            self.log.error("Unkown error: %s" % str(err))
        finally:
            # Always close the file
            self.rootgrp.close()

    def populate_global_attributes(self):
        
        # Write Global Attributes
        self.rootgrp.setncattr("title", "Orbit Data for Sea Ice Mass Reconciliation Exercise (SIMRE)")
        self.rootgrp.setncattr("project", "Arctic+ Theme 2: Sea Ice Mass")

        summary = (
            'Collection of along-track sea ice thickness estimates and statistical parameters'
            'co-located with validation datasets (calval orbit: %s) for the Sea Ice Mass Reconciliation exercise (SIMRE) ')
        summary = summary % self.ensbl.orbit_id
        summary = textwrap.dedent(summary).strip()
        
        self.rootgrp.setncattr("summary", summary)
        self.rootgrp.setncattr("orbit_id", self.ensbl.orbit_id)
        self.rootgrp.setncattr("ensemble_size_seconds", self.ensbl.member_size_seconds)
        self.rootgrp.setncattr("cdm_data_type", "Trajectory")
        self.rootgrp.setncattr("creator_name", "Stefan Hendricks")
        self.rootgrp.setncattr("creator_email", "stefan.hendricks@awi.de")

        dataset_ids = self.ensbl.dataset_ids
        self.rootgrp.setncattr("datasets", ",".join(dataset_ids))
        for dataset_id in dataset_ids:
            dataset_metadata = self.ensbl.get_dataset_metadata(dataset_id)
            for key in ["summary", "reference", "contributor", "institution", "version"]:
                attr_name = key+"_"+dataset_id
                self.rootgrp.setncattr(attr_name, getattr(dataset_metadata, key))
    
        calval_dataset_ids = []
        for calval_id in self.cv_ensbl.dataset_ids:
            source_id = calval_id[-3:]
            calval_dataset_ids.append(source_id)
        self.rootgrp.setncattr("calval_datasets", ",".join(calval_dataset_ids))
    def populate_variables(self):
            
        # Get shape of variables
        shape = np.shape(self.ensbl.time)
        dimdict = OrderedDict([("time", shape[0])])
        
        # Write dimensions
        dims = dimdict.keys()
        for key in dims:
            self.rootgrp.createDimension(key, dimdict[key])

        # Write Variables
        dim = tuple(dims)
        time = self.ensbl.time
        time_num = date2num(time, self.units, self.calendar)
        dtype_str = time_num.dtype.str
        varlon = self.rootgrp.createVariable("time", dtype_str, dim, zlib=True)
        setattr(varlon, "long_name", "utc timestamp")
        setattr(varlon, "units", self.units)
        varlon[:] = time_num

        dtype_str = self.ensbl.longitude.dtype.str
        varlon = self.rootgrp.createVariable("longitude", dtype_str, dim, zlib=True)
        setattr(varlon, "long_name", "longitude of ensemble center")
        setattr(varlon, "standard_name", "longitude")
        setattr(varlon, "units", "degrees_east")
        varlon[:] = self.ensbl.longitude

        varlat = self.rootgrp.createVariable("latitude", dtype_str, dim, zlib=True)
        setattr(varlat, "long_name", "latitude of ensemble center")
        setattr(varlat, "standard_name", "latitude")
        setattr(varlat, "units", "degrees_north")
        varlat[:] = self.ensbl.latitude

        ensemble_mean = self.ensbl.get_ensemble_mean()
        varmsit = self.rootgrp.createVariable("mean_thickness", dtype_str, dim, zlib=True)
        setattr(varmsit, "long_name", "Mean thickness from ensemble members")
        setattr(varmsit, "standard_name", "sea_ice_thickness")
        setattr(varmsit, "units", "m")
        varmsit[:] = ensemble_mean

        ensemble_min, ensemble_max = self.ensbl.get_ensemble_minmax()
        varminsit = self.rootgrp.createVariable("min_thickness", dtype_str, dim, zlib=True)
        setattr(varminsit, "long_name", "Minimum thickness of all ensemble members")
        setattr(varminsit, "standard_name", "sea_ice_thickness")
        setattr(varminsit, "units", "m")
        varminsit[:] = ensemble_min

        varmaxsit = self.rootgrp.createVariable("max_thickness", dtype_str, dim, zlib=True)
        setattr(varmaxsit, "long_name", "Maximum thickness of all ensemble members")
        setattr(varmaxsit, "standard_name", "sea_ice_thickness")
        setattr(varmaxsit, "units", "m")
        varmaxsit[:] = ensemble_max

        npts = self.ensbl.n_contributing_members
        varnpts = self.rootgrp.createVariable("n_points", npts.dtype.str, dim, zlib=True)
        setattr(varnpts, "long_name", "Number of ensemble members")
        setattr(varnpts, "units", "1")
        varnpts[:] = npts

        for calval_id in self.cv_ensbl.dataset_ids:
            dataset_mean = self.cv_ensbl.get_member_mean(calval_id)
            source_id = calval_id[-3:]
            varname = "calval_"+source_id
            cvdset = self.rootgrp.createVariable(varname, dataset_mean.dtype.str, dim, zlib=True)
            long_name = "Validation thickness (%s)" % source_id.upper()
            setattr(cvdset, "long_name", long_name)
            setattr(cvdset, "standard_name", "sea_ice_thickness")
            setattr(cvdset, "units", "m")
            cvdset[:] = dataset_mean

    @property
    def output_folder(self):
        return str(self._folder)

    @property
    def output_filepath(self):
        filename = "simre-l2-sit-reconciled-%s.nc" % (self.ensbl.orbit_id)
        return os.path.join(self.output_folder, filename)

    @property
    def ensbl(self):
        return self._ensbl

    @property
    def cv_ensbl(self):
        return self._cv_ensbl

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
        
        self.rootgrp.setncattr("summary", summary)
        self.rootgrp.setncattr("region_id", self.ensbl.region_id)
        self.rootgrp.setncattr("region_label", self.ensbl.region_label)
        self.rootgrp.setncattr("period_id", self.ensbl.period_id)
        self.rootgrp.setncattr("cdm_data_type", "Grid")
        self.rootgrp.setncattr("spatial_resolution", "25.0 km grid spacing")
        self.rootgrp.setncattr("geospatial_bounds_crs", "EPSG:6931")
        self.rootgrp.setncattr("creator_name", "Stefan Hendricks")
        self.rootgrp.setncattr("creator_email", "stefan.hendricks@awi.de")

        dataset_ids = self.ensbl.dataset_ids
        self.rootgrp.setncattr("datasets", ",".join(dataset_ids))
        for dataset_id in dataset_ids:
            dataset_metadata = self.ensbl.get_dataset_metadata(dataset_id)
            for key in ["summary", "reference", "contributor", "institution", "version"]:
                attr_name = key+"_"+dataset_id
                self.rootgrp.setncattr(attr_name, getattr(dataset_metadata, key))
    
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
