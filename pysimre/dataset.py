# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:46:20 2017

@author: shendric
"""

from pysimre.clocks import UTCTAIConverter, daycnv
from pysimre.misc import ClassTemplate, file_basename
from pysimre.proj import SIMREGridDefinition, get_region_def, TARGET_AREA_DEF

from pyresample import image, geometry

import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

from scipy.interpolate import griddata

from netCDF4 import Dataset, num2date

from dateutil.relativedelta import relativedelta
from glob import glob
import numpy as np
import os
import sys
import struct
import simplejson


# %% Classess for orbit datasets

def OrbitThicknessDataset(dataset_id, *args, **kwargs):
    """ Returns orbit data object for given orbit data id and source
    filename """

    # Get the name of the class for the dataset id
    class_map = {
            "ucl": "UCLOrbitThickness",
            "nasa_jpl": "NASAJPLOrbitThickness",
            "awi": "AWIOrbitThickness",
            "ccicdr": "CCICDROrbitThickness"}
    class_name = class_map.get(dataset_id, None)
    if class_name is None:
        msg = "Orbit dataset class not implemented for id: %s"
        raise ValueError(msg % str(dataset_id))

    # Return the Dataset class
    return globals()[class_name](*args, **kwargs)


class OrbitThicknessBaseClass(object):

    # define data types of parameters (default: single precision float)
    dtype = defaultdict(lambda: "f4", time=object)

    def __init__(self, track_id="n/a", orbit="n/a"):
        self.filename = None
        self.track_id = track_id
        self.orbit = orbit
        self.time = None
        self.longitude = None
        self.latitude = None
        self.sea_ice_thickness = None
        self.ice_density = None
        self.snow_depth = None
        self.snow_density = None

    def init_parameter_groups(self, n_records, parameter_list):
        """ Create arrays for given parameters """
        for name in parameter_list:
            array = np.ndarray(shape=n_records, dtype=self.dtype[name])
            setattr(self, name, array)

    def clip_to_latbox(self, lat_limits, direction):
        """ Computes the overlap of data points for a given latitude box
        [lat_min, lat_max] in either the 'ascending' or 'descending'
        direction of the orbit and clips all parameters accordingly
        """

        if direction not in ["ascending", "descending"]:
            msg = "direction argument must be \'ascending\' or \'descending\'"
            raise ValueError(msg)

        # Compute latitude changes (assume continuity for first entry)
        lat_delta = np.ediff1d(self.latitude)
        lat_delta = np.insert(lat_delta, 0, lat_delta[0])

        # Condition: orbit direction
        if direction == "ascending":
            condition_direction = lat_delta > 0.0
        else:
            condition_direction = lat_delta <= 0.0

        # Condition: in latbox
        condition_latbox = np.logical_and(
                self.latitude >= lat_limits[0],
                self.latitude <= lat_limits[1])

        # get Indices
        condition = np.logical_and(condition_direction, condition_latbox)
        indices = np.where(condition)[0]

        # Sanity check
        if len(indices) == 0:
            warnings.warn("No points in latbox, clipping all points")

        self.clip_to_indices(indices)

    def wraplons(self):
        lon_orig = self.longitude
        self.longitude = np.mod(lon_orig+180., 360.) - 180.

    def clip_to_indices(self, indices):
        """ Create a subset of all parameters for the given list of indices """
        for parameter_name in self.parameter_list:
            data = getattr(self, parameter_name)
            setattr(self, parameter_name, data[indices])

    def get_ensemble_items(self, time_range):
        """ Returns points for orbit ensemble statistics """
        in_ensemble_item = np.logical_and(
                self.time > time_range[0],
                self.time <= time_range[2])
        indices = np.where(in_ensemble_item)[0]
        return self.longitude[indices], self.latitude[indices], \
            self.sea_ice_thickness[indices]

    @property
    def time_range(self):
        valid_indices = np.where(np.isfinite(self.sea_ice_thickness))[0]
        return [self.time[valid_indices[0]], self.time[valid_indices[-1]]]

    @property
    def n_records(self):
        return len(self.longitude)

    @property
    def has_timestamp(self):
        return type(self.time) is np.ndarray

    def __repr__(self):
        output = "\npySIMRE Orbit Thickness object:\n"
        output += "%16s: %s\n" % ("dataset_id", self.dataset_id)
        output += "%16s: %s\n" % ("n_records", self.n_records)
        output += "%16s: %s" % ("orbit_id", self.orbit)
        return output


class NASAJPLOrbitThickness(OrbitThicknessBaseClass):

    # Metatadata
    dataset_id = "nasa_jpl"
    dataset_label = "NASA-JPL"

    # Has the following parameters
    parameter_list = ["time", "longitude", "latitude", "ice_density",
                      "snow_density", "snow_depth", "sea_ice_thickness"]

    # Parameter properties
    ice_density_is_fixed = True
    snow_density_is_fixed = False

    # Physical parameters
    default_ice_density = 920.0
    water_density = 1024.0

    # Housekeeping
    n_header_lines = 3

    def __init__(self, filename, **kwargs):

        super(NASAJPLOrbitThickness, self).__init__(**kwargs)
        self.filename = filename
        self.parse_filename()
        self.parse()
        self.wraplons()

    def parse_filename(self):
        basename = file_basename(self.filename)
        strarr = basename.split("_")
        self.orbit = strarr[3]

    def parse(self):
        """ Read the file """

        # read the content of the file
        with open(self.filename, "r") as fh:
            content = fh.readlines()

        # remove header
        content = content[self.n_header_lines:]

        # Create the parameter groups
        n_records = len(content)
        self.init_parameter_groups(n_records, self.parameter_list)

        tai2utc = UTCTAIConverter()

        for i, line in enumerate(content):

            array = [float(s) for s in line.split()]

            # compute timestamp
            days = int(array[1])
            seconds = int(array[2])
            musecs = int(1e6*(array[2]-seconds))
            tai_dt = datetime(int(array[0]), 1, 1) + \
                timedelta(days=days-1, seconds=seconds, microseconds=musecs)

            self.time[i] = tai2utc.tai2utc(tai_dt)[0]

            # Transfer data
            self.latitude[i] = array[3]
            self.longitude[i] = array[4]
            self.sea_ice_thickness[i] = array[5]
            self.snow_depth[i] = array[6]
            self.snow_density[i] = array[7]

        # fill ice density array with default values
        self.ice_density[:] = self.default_ice_density


class AWIOrbitThickness(OrbitThicknessBaseClass):

    # Metadata
    dataset_id = "awi"
    config_file_name = "cryosat_seaice_proc_config.json"

    # Has the following parameters
    parameter_list = ["time", "longitude", "latitude", "ice_density",
                      "snow_density", "snow_depth", "sea_ice_thickness"]

    # Parameter properties
    ice_density_is_fixed = False
    snow_density_is_fixed = False

    # Housekeeping
    header_size_bytes = 106
    datagroup_byte_size = 8

    def __init__(self, filename, **kwargs):

        super(AWIOrbitThickness, self).__init__(**kwargs)
        self.filename = filename
        self.file_def = configuration_file_ordereddict(self.config_filename)
        self.parse_filename()
        self.parse_content()
        self.construct_data_groups()
        self.wraplons()

    def parse_filename(self):
        """
        Gets Meta Information from filename
        Example filename:
            CS2_021093_20140401T003839_20140401T004237_B001_AWIPROC01.dat
        """
        fbase = file_basename(self.filename)
        strarr = fbase.split('_')
        # Orbit Number
        self.orbit = np.int16(strarr[1])

    def parse_content(self):
        # Read the Header
        self._parse_header()
        # Calculate the number of records
        self._get_n_records()
        # Construct the data groups
        self._construct_datagroups()
        # Parse file content into data array
        self._parse_content()
        # Create attribute for each parameter
        self._create_attributes()

    def construct_data_groups(self):
        datenum = self.get_datagroup("time")
        timestamp = [self._get_datetime(time) for time in datenum]
        self.time = np.array(timestamp)
        self.longitude = self.get_datagroup("lon")
        self.latitude = self.get_datagroup("lat")
        self.ice_density = self.get_datagroup("rho_i")
        self.snow_density = self.get_datagroup("rho_s")
        self.snow_depth = self.get_datagroup("sd")
        self.sea_ice_thickness = self.get_datagroup("sit")

    def get_datagroup(self, tag):
        index = self.registered_datagroups[tag]
        return np.squeeze(self.data[:, index])

    def _parse_header(self):
        with open(self.filename, 'rb') as f:
            header = f.read(self.header_size_bytes)
        self.lon_limit = struct.unpack('<2f', header[0:8])
        self.lat_limit = struct.unpack('<2f', header[8:16])
        self.content_flags = np.array(struct.unpack('<45h', header[16:106]))

    def _parse_content(self):
        struct_record_parser = self._get_struct_record_parser()
        with open(self.filename, 'rb') as f:
            # read header information again (-> Nirvana)
            f.read(self.header_size_bytes)
            # Read full file content
            for i in np.arange(self.nrecs):
                record = f.read(self.record_byte_size)
                self.data[i, :] = struct.unpack(struct_record_parser, record)
        self._is_in_roi = np.full(self.nrecs, True)

    def _create_attributes(self):
        for parameter_name in self.registered_datagroups.keys():
            setattr(self, parameter_name, self.get_datagroup(parameter_name))

    def _get_n_records(self):
        self._file_size = os.path.getsize(self.filename)
        self.n_datagroups = len(self._get_datagroup_indices())
        self.nrecs = (self._file_size - self.header_size_bytes) / (
            self.n_datagroups * self.datagroup_byte_size)
        self.record_byte_size = self.n_datagroups * self.datagroup_byte_size

    def _construct_datagroups(self):
        self.registered_datagroups = {}
        datagroup_keys = list(self.file_def["output"])
        for i, index in enumerate(self._get_datagroup_indices()):
            self.registered_datagroups[datagroup_keys[index]] = i
        self.data = np.ndarray(
            shape=(self.nrecs, self.n_datagroups),
            dtype=np.float64)
        self._in_export = np.ones(shape=(self.nrecs), dtype=np.bool)

    def _get_datagroup_indices(self):
        return np.where(self.content_flags.astype(bool))[0]

    def _get_struct_record_parser(self):
        return "<%gd" % self.n_datagroups

    def _get_datetime(self, xjd):
        year, month, day, hours = daycnv(xjd)
        timestamp = datetime(year, month, day) + relativedelta(hours=hours)
        return timestamp

    def _get_datetime_str(self, datetime):
        datetime_str = '%sT%s' % (
            datetime.strftime('%Y%m%d'),
            datetime.strftime('%H%M%S'))
        return datetime_str

    def _get_datetime_records_str(self, datetime):
        datetime_str = '%sT%s.%03d' % (
            datetime.strftime('%Y-%m-%d'),
            datetime.strftime('%H:%M:%S'),
            datetime.microsecond/1000)
        return datetime_str

    @property
    def config_filename(self):
        directory = os.path.split(self.filename)[0]
        return os.path.join(directory, self.config_file_name)


class UCLOrbitThickness(OrbitThicknessBaseClass):

    # Metadata
    dataset_id = "ucl"

    epoch = datetime(1950, 1, 1)

    # Has the following parameters
    parameter_list = ["time", "longitude", "latitude",
                      "sea_ice_thickness"]

    def __init__(self, filename, **kwargs):
        super(UCLOrbitThickness, self).__init__(**kwargs)
        self.filename = filename
        self.parse()
        self.wraplons()

    def parse(self):
        """ Parse data content """

        # Parse the entire file
        with open(self.filename, "r") as fh:
            content = fh.readlines()

        # Init data groups
        n_records = len(content)
        self.init_parameter_groups(n_records, self.parameter_list)

        tai2utc = UTCTAIConverter()

        # Parse data content
        for i, line in enumerate(content):
            strarr = line.split()

            # get timestamp
            days = float(strarr[2])
            tai_dt = self.epoch + relativedelta(days=days)
            self.time[i] = tai2utc.tai2utc(tai_dt)[0]

            # geolocation parameters
            self.longitude[i] = float(strarr[4])
            self.latitude[i] = float(strarr[3])

            # thickness: respect is valid flag
            sit = float(strarr[6]) if int(strarr[5]) == 1 else np.nan
            self.sea_ice_thickness[i] = sit


class CCICDROrbitThickness(OrbitThicknessBaseClass):

    # Metadata
    dataset_id = "ccicdr"

    # Has the following parameters
    parameter_list = ["time", "longitude", "latitude",
                      "sea_ice_thickness", "snow_depth", "snow_density",
                      "ice_density"]

    parameter_map = {
            "time": "timestamp",
            "longitude": "longitude",
            "latitude": "latitude",
            "sea_ice_thickness": "sea_ice_thickness",
            "snow_depth": "snow_depth",
            "snow_density": "snow_density",
            "ice_density": "ice_density"}

    time_units = "seconds since 1970-01-01"
    time_calendar = "standard"

    def __init__(self, filename, **kwargs):
        super(CCICDROrbitThickness, self).__init__(**kwargs)
        self.filename = filename
        self.parse()
        self.wraplons()

    def parse(self):
        data = ReadNC(self.filename,
                      convert2datetime=["timestamp"],
                      time_units=self.time_units,
                      time_calendar=self.time_calendar)
        for parameter_name in self.parameter_list:
            nc_parameter_name = self.parameter_map[parameter_name]
            setattr(self, parameter_name, getattr(data, nc_parameter_name))

# %% Classes for gridded datasets


def GridSourceData(class_name, filepath, repo_dir, *ids, **kwargs):
    """ Getter function for the gridded source data. *ids is the
    array [dataset_id, region_id, period_id] """

    try:
        dataset = globals()[class_name](filepath, repo_dir, *ids, **kwargs)
    except KeyError:
        print "Unkown class in pysimre.dataset: %s" % str(class_name)
        sys.exit()

    # Return the Dataset class
    return dataset


class SourceGridBaseClass(ClassTemplate):

    def __init__(self, filename, repo_dir, dataset_id, region_id, period_id,
                 apply_land_mask=True, region_label=None):
        super(SourceGridBaseClass, self).__init__(self.__class__.__name__)

        # Source metadata
        self.filename = filename
        self.period_id = period_id
        self.region_id = region_id
        self.region_label = region_label
        self.dataset_id = dataset_id
        self.repo_dir = repo_dir
        self.apply_land_mask = apply_land_mask
        self.source_longitude = None
        self.source_latitude = None
        self.source_thickness = None

        # Commm grid definition
        self.griddef = SIMREGridDefinition()
        lon, lat = self.griddef.get_grid_coordinates()
        self.longitude = lon
        self.latitude = lat
        self.thickness = None

    def extract_test_region(self, region_id):
        """ Returns a data object for a given region """
        region_data = RegionGrid(self.repo_dir, region_id, region_label=self.region_label)
        region_data.from_source_grid(self)
        return region_data

    def resample_sourcegrid_to_targetgrid(self):
        """ Use pyresample for nearest neighbour resampling for gridded source
        data """

        # Resample the image
        resample_container = image.ImageContainerNearest(
            self.source_thickness, self.source_areadef,
            radius_of_influence=25000, fill_value=None)
        resample_result = resample_container.resample(TARGET_AREA_DEF)

        # pyresample uses masked arrays -> set nan's to missing data
        data_ease2 = resample_result.image_data
        data_ease2[np.where(data_ease2.mask)] = np.nan
        self.thickness = data_ease2

    def resample_sourcepoints_to_targetgrid(self):
        """ Use gridding to populate the target grid with gridded source
        data points given as a list of lon, lat, thickness """

        # Do gridding in projection coordinates to avoid issues
        # with longitude wrapping
        source_x, source_y = self.griddef.get_projection_coordinates(
                self.source_longitude, self.source_latitude)

        target_x, target_y = self.griddef.get_projection_coordinates(
                self.longitude, self.latitude)

        # Grid the data usind nearest neighbour interpolation
        # NOTE: griddata will fill the entire grid. Thus, if only
        #       source data for a given region is defined, the grid
        #       will contain garbage for all grid cells outside the region
        #       but we do not want to interpolate over data gaps/lands
        griddata_settings = {'method': 'nearest', 'rescale': False}
        target_thickness = griddata((source_x, source_y),
                                    self.source_thickness,
                                    (target_x, target_y),
                                    **griddata_settings)

        # Mask out empty target grid cells by computing an array
        # that is 1.0 for a grid cell that has a target point in it and
        # nan otherwise
        nodata_mask = np.full(target_thickness.shape, np.nan)
        x_indices, y_indices = self.griddef.grid_indices(
                self.source_longitude, self.source_latitude)
        for x_index, y_index in zip(x_indices, y_indices):
            nodata_mask[y_index, x_index] = 1.0
        target_thickness *= nodata_mask

        # Set gridded source thickness
        self.thickness = target_thickness

    def grid_sourcepoints_to_targetgrid(self):
        """ Use gridding to populate the target grid with orbit source data
        points (lon, lat, thickness) """

        # Sort all data points into stack for each grid cell
        stack = self.parameter_stack

        # Get the grid indices for each point in the source input data
        xi, yj = self.griddef.grid_indices(
                self.source_longitude, self.source_latitude)

        # Sort all points in the stack
        j = 0
        for i in np.arange(len(self.source_longitude)):
            x, y = int(xi[i]), int(yj[i])
            try:
                stack[y][x].append(self.source_thickness[i])
            except IndexError:  # orbit point outside grid -> ignore
                j += 1
                pass

        # Compute gridded thickness as mean for each stack entry
        thickness = np.full(self.longitude.shape, np.nan)
        for xi in self.grid_xi_range:
            for yj in self.grid_yj_range:
                stacked_data = np.array(stack[yj][xi])
                thickness[yj, xi] = np.nanmean(stacked_data)

        self.thickness = thickness

    def apply_target_grid_masks(self, debug=False):
        """ Apply a fixed mask for the target grid. In the moment this
        is only a land sea mask to prevent issues with data being interpolated
        to areas with land. """

        # Do nothing if flag not set
        if not self.apply_land_mask:
            return

        # Get the name of teh land mask grid file:
        mask_filename = "landsea_"+self.griddef.grid_id+".nc"
        mask_path = os.path.join(self.repo_dir, "masks", mask_filename)

        # Read the mask file
        # Note: the flag in the land mask file is 0: sea, 1: mixed, 2: land
        landmask = ReadNC(mask_path)

        # Apply the mask
        # (do not account for pixel with land or partial land cover)
        self.thickness[np.where(landmask.mask > 0)] = np.nan

        if debug:

            import matplotlib.pyplot as plt

            plt.figure("thickness")
            plt.imshow(self.thickness)

            plt.figure("landmask")
            plt.imshow(landmask.mask)

            plt.show()
            stop

    @property
    def parameter_stack(self):
        """ Returns a stack for source orbit data """
        dimx, dimy = self.griddef.extent.numx, self.griddef.extent.numy
        return [[[] for _ in range(dimx)] for _ in range(dimy)]

    @property
    def grid_xi_range(self):
        return np.arange(self.griddef.extent.numx)

    @property
    def grid_yj_range(self):
        return np.arange(self.griddef.extent.numy)

    @property
    def source_filename(self):
        if isinstance(self.filename, list):
            return [os.path.split(f)[-1] for f in self.filename]
        else:
            return os.path.split(self.filename)[-1]


class RegionGrid(ClassTemplate):

    def __init__(self, repo_dir, region_id, region_label=""):
        super(RegionGrid, self).__init__(self.__class__.__name__)
        self._repo_dir = repo_dir
        self._region_id = region_id
        self._region_label = region_label
        self._dataset_id = None
        self._period_id = None

    def from_source_grid(self, source_grid):
        """ Extract region data from source grid
        (needs to be reference grid) """
        self._source_grid = source_grid
        self._dataset_id = source_grid.dataset_id
        self._period_id = source_grid.period_id
        self._source_filename = source_grid.source_filename
        self.crop_to_region()

    def crop_to_region(self):
        """ Get the subset from the source grid """

        # Get the definition
        region_def = get_region_def(self.repo_dir, self.region_id)

        # Get indices in source grid that match region
        source = self._source_grid
        in_lon_range = np.logical_and(
                source.longitude >= region_def.longitude_limits[0],
                source.longitude <= region_def.longitude_limits[1])
        in_lat_range = np.logical_and(
                source.latitude >= region_def.latitude_limits[0],
                source.latitude <= region_def.latitude_limits[1])
        in_region = np.logical_and(in_lon_range, in_lat_range)
        region_indices = np.where(in_region)
        outside_indices = np.where(np.logical_not(in_region))

        # Get indices ranges
        i0, i1 = np.amin(region_indices[0]), np.amax(region_indices[0])
        j0, j1 = np.amin(region_indices[1]), np.amax(region_indices[1])

        # Extract longitude/latitude/thickness for region subset
        region_longitude = np.copy(source.longitude)
        region_longitude = region_longitude[i0:i1, j0:j1]

        region_latitude = np.copy(source.latitude)
        region_latitude = region_latitude[i0:i1, j0:j1]

        region_thickness = np.copy(source.thickness)
        region_thickness[outside_indices] = np.nan
        region_thickness = region_thickness[i0:i1, j0:j1]

        # Set to object
        self.longitude = region_longitude
        self.latitude = region_latitude
        self.thickness = region_thickness

    def write_to_netcdf(self, output_filepath):
        """ Writes region data to netcdf. Automatically creates path
        and filename by default. """

        # Get metadata
        shape = np.shape(self.longitude)
        dimdict = OrderedDict([("lat", shape[0]), ("lon", shape[1])])
        if self.source_filename is list:
            source_filename = ", ".join(self.source_filename)
        else:
            source_filename = self.source_filename

        lons, lats = self.longitude, self.latitude

        self.log.info("Write SIMRE netcdf: %s" % output_filepath)
        # Open the file
        rootgrp = Dataset(output_filepath, "w")

        # Write Global Attributes
        rootgrp.setncattr("title", "Regional Data for Sea Ice Mass " +
                          "Reconciliation Exercise (SIMRE)")
        rootgrp.setncattr("project", "Arctic+ Theme 2: Sea Ice Mass")
        rootgrp.setncattr("source", source_filename)
        rootgrp.setncattr("dataset_id", self.dataset_id)
        rootgrp.setncattr("region_id", self.region_id)
        rootgrp.setncattr("region_label", self.region_label)
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
        rootgrp.close()

    def from_netcdf(self, netcdf_filename):
        """ Populate data groups from SIMRE netcdf """

        # Read the data
        data = ReadNC(netcdf_filename)

        # Transfer ids
        region_id = data.region_id
        if region_id != self.region_id:
            msg = "Region id in SIMRE netcdf (%s) not Region id of object (%s)"
            msg = msg % (region_id, self.region_id)
            self.error("region_id-mismatch", msg)
            self.error.raise_on_error()
        self._dataset_id = data.dataset_id
        self._period_id = data.period_id

        self.longitude = data.longitude
        self.latitude = data.latitude
        self.thickness = data.sea_ice_thickness

    @property
    def region_id(self):
        return str(self._region_id)

    @property
    def region_label(self):
        return str(self._region_label)

    @property
    def dataset_id(self):
        return str(self._dataset_id)

    @property
    def period_id(self):
        return str(self._period_id)

    @property
    def source_filename(self):
        return str(self._source_filename)

    @property
    def repo_dir(self):
        return self._repo_dir


class AWIGridThickness(SourceGridBaseClass):

    # Area definition for pyresample
    source_areadef = geometry.AreaDefinition(
        'ease2', 'NSIDC North Polar Stereographic (25km)',
        'ease2',
        {'lat_0': '90.00', 'lat_ts': '70.00',
         'lon_0': '0.00', 'proj': 'laea'},
        720, 720,
        [-9000000., -9000000., 9000000., 9000000.])

    def __init__(self, *args, **kwargs):
        super(AWIGridThickness, self).__init__(*args, **kwargs)
        self.read_nc()
        self.resample_sourcegrid_to_targetgrid()
        self.apply_target_grid_masks()

    def read_nc(self):
        data = ReadNC(self.filename)
        self.source_thickness = np.flipud(data.sea_ice_thickness)


class CCICDRGridThickness(SourceGridBaseClass):

    def __init__(self, *args, **kwargs):
        super(CCICDRGridThickness, self).__init__(*args, **kwargs)
        self.read_nc()
        self.apply_target_grid_masks()

    def read_nc(self):
        data = ReadNC(self.filename)
        self.source_thickness = data.sea_ice_thickness[0, :, :]
        self.thickness = data.sea_ice_thickness[0, :, :]


class CS2SMOSGridThickness(SourceGridBaseClass):

    # Area definition for pyresample
    source_areadef = geometry.AreaDefinition(
        'ease2', 'NSIDC North Polar Stereographic (25km)',
        'ease2',
        {'lat_0': '90.00', 'lat_ts': '70.00',
         'lon_0': '0.00', 'proj': 'laea'},
        720, 720,
        [-9000000., -9000000., 9000000., 9000000.])

    def __init__(self, *args, **kwargs):
        super(CS2SMOSGridThickness, self).__init__(*args, **kwargs)
        self.read_nc()
        self.resample_sourcegrid_to_targetgrid()
        self.apply_target_grid_masks()

    def read_nc(self):
        thickness_grids = []
        n_files = len(self.filename)
        for filename in self.filename:
            data = ReadNC(filename)
            thickness_grids.append(data.analysis_thickness)

        average_thickness = thickness_grids[0]
        for i in np.arange(1, n_files):
            average_thickness += thickness_grids[i]
        average_thickness /= np.float(n_files)
        self.source_thickness = np.flipud(average_thickness)


class LEGOSGridThickness(SourceGridBaseClass):

        # Area definition for pyresample
    source_areadef = geometry.AreaDefinition(
        'ease2', 'NSIDC North Polar Stereographic (12.5km)',
        'ease2',
        {'lat_0': '90.00', 'lat_ts': '70.00',
         'lon_0': '0.00', 'proj': 'laea'},
        722, 722,
        [-4512500., -4512500., 4512500., 4512500.])

    def __init__(self, *args, **kwargs):
        super(LEGOSGridThickness, self).__init__(*args, **kwargs)
        self.read_nc()
        self.resample_sourcegrid_to_targetgrid()
        self.apply_target_grid_masks()

    def read_nc(self):
        data = ReadNC(self.filename)
        self.source_thickness = np.flipud(data.sea_ice_thickness)


class NasaGSFCGridThickness(SourceGridBaseClass):

        # Area definition for pyresample
    source_areadef = geometry.AreaDefinition(
        'nsidc_npstere', 'NSIDC North Polar Stereographic (25km)',
        'nsidc_npstere',
        {'lat_0': '90.00', 'lat_ts': '70.00',
         'lon_0': '-45.00', 'proj': 'stere'},
        304, 448,
        [-3800000.0, -5350000.0, 3800000.0, 5850000.0])

    def __init__(self, *args, **kwargs):
        super(NasaGSFCGridThickness, self).__init__(*args, **kwargs)
        self.read_nc()
        self.resample_sourcegrid_to_targetgrid()
        self.thickness = np.flipud(self.thickness)
        self.apply_target_grid_masks()

    def read_nc(self):
        data = ReadNC(self.filename)

        sit = data.sea_ice_thickness
        sit[np.where(sit < -999.)] = np.nan
        self.source_thickness = sit  # np.flipud(sit)


class NasaJPLGridThickness(SourceGridBaseClass):

    def __init__(self, *args, **kwargs):
        super(NasaJPLGridThickness, self).__init__(*args, **kwargs)
        self.read_ascii()
        if not self.error.status:
            self.resample_sourcepoints_to_targetgrid()
            self.apply_target_grid_masks()

    def read_ascii(self):
        # NASA JPL data comes as one ascii file per region/period
        # The filename argument therefore is a search string that should
        # return 1 file
        try:
            actual_filename = glob(self.filename)[0]
        except IndexError:  # no file exists
            msg = "No input files found for %s.%s.%s [%s]" % (
                    self.dataset_id, self.region_id, self.period_id,
                    self.filename)
            self.log.warning(msg)
            self.error.add_error("missing-file(s)", msg)
            return

        # Read the daata
        col_names = ["latitude", "longitude", "thickness"]
        data = np.genfromtxt(actual_filename, names=col_names)
        self.source_longitude = data["longitude"]
        self.source_latitude = data["latitude"]
        self.source_thickness = data["thickness"]


class UCLGridThickness(SourceGridBaseClass):

    def __init__(self, *args, **kwargs):
        super(UCLGridThickness, self).__init__(*args, **kwargs)
        self.read_ascii()
        if not self.error.status:
            self.grid_sourcepoints_to_targetgrid()
            self.apply_target_grid_masks()

    def read_ascii(self):
        # UCL data comes as one ascii file per day for each region/period
        # The filename argument therefore is a search string that should
        # return several files, which content needs to concatenated and
        filenames = glob(self.filename)
        if len(filenames) == 0:
            msg = "No input files found for %s.%s.%s [%s]" % (
                    self.dataset_id, self.region_id, self.period_id,
                    self.filename)
            self.log.warning(msg)
            self.error.add_error("missing-file(s)", msg)
            return

        # Parse and stack the content
        col_names = ["time", "longitude", "latitude", "thickness", "flag",
                     "concentration"]
        longitude, latitude, thickness = [], [], []
        for filename in filenames:
            # Read the data
            data = np.genfromtxt(filename, names=col_names)
            longitude.extend(data["longitude"])
            latitude.extend(data["latitude"])
            thickness.extend(data["thickness"])

        self.source_longitude = np.array(longitude)
        self.source_latitude = np.array(latitude)
        self.source_thickness = np.array(thickness)


# %% Classes for cal/val datasets


def CalValDataset(class_name, filepath, dataset_id, orbit_id, metadata):
    """ Returns the native data object for the cal/val dataset """
    return globals()[class_name](filepath, dataset_id, orbit_id, metadata)


class CryoValCSV(OrbitThicknessBaseClass):
    """ Container for CryoVAL-SI csv datasets """

    parameter_list = ["time", "longitude", "latitude", "sea_ice_thickness"]
    parameter_map = {"longitude": "Longitude", "latitude": "Latitude"}

    def __init__(self, filepath, dataset_id, orbit_id, metadata):
        super(CryoValCSV, self).__init__(orbit=orbit_id, track_id=dataset_id)
        self.dataset_id = dataset_id
        self._parameter_name = metadata.parameter_name
        self._parameter_target = metadata.parameter_target
        self._filename = filepath
        self.label = metadata.label

        self._parse()

    def _parse(self):
        """ Read the csv file and only save time/lon/lat/target thickness """

        # Parse content
        data = np.genfromtxt(self._filename, delimiter=',', names=True)

        # Init dataset object
        n_records = np.shape(data)[0]
        self.init_parameter_groups(n_records, self.parameter_list)

        # Transfer time
        y, mo, d = data["Year"], data["Month"], data["Day"]
        h, mi, s = data["Hour"], data["Minute"], data["Second"]

        for i in np.arange(self.n_records):
            self.time[i] = datetime(
                    int(y[i]), int(mo[i]), int(d[i]),
                    int(h[i]), int(mi[i]))
            self.time[i] += relativedelta(seconds=s[i])

        # Transfer lon/lat
        for name in ["longitude", "latitude"]:
            setattr(self, name, data[self.parameter_map[name]])

        # Transfer target parameter
        setattr(self, self._parameter_target, data[self._parameter_name])


# %% General support classes & functions


class ReadNC(object):
    """ Simple to parse content of netCDF file into a python object
    with global attributes and variables """

    def __init__(self, filename, verbose=False, autoscale=True,
                 nan_fill_value=False, time_units="seconds since 1970-01-01",
                 time_calendar="standard", convert2datetime=[]):
        self.time_units = time_units
        self.time_calendar = time_calendar
        self.convert2datetime = convert2datetime
        self.parameters = []
        self.attributes = []
        self.verbose = verbose
        self.autoscale = autoscale
        self.nan_fill_value = nan_fill_value
        self.filename = filename
        self.parameters = []
        self.read_content()

    def read_content(self):
        self.keys = []
        f = Dataset(self.filename)
        f.set_auto_scale(self.autoscale)

        # Get the global attributes
        for attribute_name in f.ncattrs():

            self.attributes.append(attribute_name)
            attribute_value = getattr(f, attribute_name)

            # Convert timestamps back to datetime objects
            if attribute_name in ["start_time", "stop_time"]:
                attribute_value = num2date(
                    attribute_value, self.time_units,
                    calendar=self.time_calendar)
            setattr(self, attribute_name, attribute_value)

        # Get the variables
        for key in f.variables.keys():
            variable = f.variables[key][:]

            try:
                is_float = variable.dtype in ["float32", "float64"]
                has_mask = hasattr(variable, "mask")
            except:
                is_float, has_mask = False, False

            if self.nan_fill_value and has_mask and is_float:
                is_fill_value = np.where(variable.mask)
                variable[is_fill_value] = np.nan

            if key in self.convert2datetime:
                variable = num2date(
                    variable, self.time_units,
                    calendar=self.time_calendar)

            setattr(self, key, variable)
            self.keys.append(key)
            self.parameters.append(key)
            if self.verbose:
                print key
        self.parameters = f.variables.keys()
        f.close()


def configuration_file_ordereddict(filename):
    """
    Read a json file and returns its contents

    Args:
        filename (str):
            Full path to configuration file

    Returns (ordereddict):
        Content of configuration files.
    """
    with open(filename, 'r') as filehandle:
        data = simplejson.load(filehandle, object_pairs_hook=OrderedDict)
    return data
