# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:46:20 2017

@author: shendric
"""

from pysimre.clocks import UTCTAIConverter, daycnv
from pysimre.misc import ClassTemplate, file_basename


import warnings
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta

from netCDF4 import Dataset, num2date

from dateutil.relativedelta import relativedelta
import numpy as np
import os
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
