# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 17:02:56 2017

@author: shendric
"""

from treedict import TreeDict
from logbook import Logger, StreamHandler
from datetime import datetime
from netCDF4 import Dataset
import numpy as np
import yaml
import sys
import os


def stdout_logger(name):
    StreamHandler(sys.stdout).push_application()
    log = Logger(name)
    return log


class ClassTemplate(object):
    """ A default class providing logging to stdout and simple
    error handling capability """

    def __init__(self, name):
        self.error = ErrorStatus(caller_id=name)
        self.log = stdout_logger(name)

    def info(self, *args, **kwargs):
        self.log.info(*args, **kwargs)

    def debug(self, *args, **kwargs):
        self.log.debug(*args, **kwargs)


class ErrorStatus(object):
    """ A custom class for error reporting """

    def __init__(self, caller_id=""):
        self.caller_id = caller_id
        self.reset()

    def add_error(self, code, message):
        """ Add an error. Error code and messages are arbitrary """
        self.status = True
        self.codes.append(code)
        self.messages.append(message)

    def raise_on_error(self):
        """ print error messages and exit program on existing error(s) """
        if self.status:
            output = "%s Critical Error(s): (%g)\n" % (
                self.caller_id, len(self.codes))
            for i in range(len(self.codes)):
                output += "  [%s] %s" % (self.codes[i], self.messages[i])
                output += "\n"
            print output
            sys.exit(1)

    def get_all_messages(self):
        output = []
        if self.status:
            for i in range(len(self.codes)):
                error_message = "%s error: [%s] %s" % (
                    self.caller_id, self.codes[i], self.messages[i])
                output.append(error_message)
        return output

    def reset(self):
        """ Remove all error messages and set to clean status """
        self.status = False
        self.codes = []
        self.messages = []

    @property
    def message(self):
        return ",".join(self.messages)


class ReadNC(object):
    """
    Quick & dirty method to parse content of netCDF file into a python object
    with attributes from file variables
    """
    def __init__(self, filename, verbose=False, autoscale=True,
                 nan_fill_value=False):
        self.error = ErrorStatus()
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

        # Open the file
        try:
            f = Dataset(self.filename)
        except RuntimeError:
            msg = "Cannot option netCDF file: %s" % self.filename
            self.error.add_error("nc-runtime-error", msg)
            self.error.raise_on_error()

        f.set_auto_scale(self.autoscale)

        # Get the global attributes
        for attribute_name in f.ncattrs():

            self.attributes.append(attribute_name)
            attribute_value = getattr(f, attribute_name)

            # Convert timestamps back to datetime objects
            # TODO: This needs to be handled better
            # if attribute_name in ["start_time", "stop_time"]:
            #     attribute_value = num2date(
            #         attribute_value, self.time_def.units,
            #         calendar=self.time_def.calendar)
            # setattr(self, attribute_name, attribute_value)

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

            setattr(self, key, variable)
            self.keys.append(key)
            self.parameters.append(key)
            if self.verbose:
                print key
        self.parameters = f.variables.keys()
        f.close()

def parse_config_file(filename, output="treedict"):
    """
    Parses the contents of a configuration file in .yaml format
    and returns the content in various formats

    Arguments:
        filename (str)
            path the configuration file

    Keywords:
        output (str)
            "treedict" (default): Returns a treedict object
            "dict": Returns a python dictionary
    """
    with open(filename, 'r') as f:
        content_dict = yaml.load(f)

    if output == "treedict":
        return TreeDict.fromdict(content_dict, expand_nested=True)
    else:
        return content_dict


def file_basename(filename, fullpath=False):
    """
    Returns the filename without file extension of a give filename (or path)
    """
    strarr = os.path.split(filename)
    file_name = strarr[-1]
    basename = file_name.split(".")[0]
    if fullpath:
        basename = os.path.join(strarr[0], basename)
    # XXX: Sketchy, needs better solution (with access to os documentation)
    return basename


def pid2dt(period_ids, center=True, numerical=False):
    """ concerts period_id(s) [yyyy-mm] into a datetime object or a numerical
    representation of the datetime """
    day = 15
    dt = []
    for period_id in period_ids:
        year, month = int(period_id[0:4]), int(period_id[6:8])
        dt.append(datetime(year, month, day))
    return dt