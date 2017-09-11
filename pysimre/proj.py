# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 09:33:24 2017

@author: Stefan
"""

from pysimre import LOCAL_PATH_RESOURCES
from pysimre.misc import ClassTemplate, parse_config_file

from pyresample import image, geometry

from treedict import TreeDict
from pyproj import Proj

import numpy as np
import os

DEFAULT_GRID_DEFINITION = "nh25kmEASE2.yaml"

# for pyresample
TARGET_AREA_DEF = geometry.AreaDefinition(
    'nh25kmease2', 'EASE2 North (25km)', 'nh25kmease2',
    {'lat_0': '90.00', 'lat_ts': '70.00',
     'lon_0': '0.00', 'proj': 'laea',
     'ellps': 'WGS84', 'datum': 'WGS84'},
    432, 432,
    [-5400000.0, -5400000.0, 5400000.0, 5400000.0])


def get_target_grid():
    """ Return longitude/latitude grid coordinates for the SIMRE default
    grid """
    grid = DefaultGridDefinition()
    return grid.get_grid_coordinates()


class DefaultGridDefinition(ClassTemplate):
    """ A container class for geospatial grids.  The main components are
    a) the projection (based on pyproj) and b) the extent and grid size.
    The class is currently designed for stereographic/equal area projection
    types. """

    def __init__(self, preset=None):
        super(DefaultGridDefinition, self).__init__(self.__class__.__name__)
        self._preset = preset
        self._metadata = {"grid_id": "n/a", "grid_tag": "n/a",
                          "hemisphere": "n/a", "resolution_tag": "n/a"}
        self._proj = None
        self._proj_dict = {}
        self._extent_dict = {}
        default_grid_definition = os.path.join(
                LOCAL_PATH_RESOURCES,
                DEFAULT_GRID_DEFINITION)
        self.set_from_griddef_file(default_grid_definition)

    def set_from_griddef_file(self, filename):
        """ Initialize the object with a grid definition (.yaml) file.
        Examples can be found in pysiral/settings/griddef """
        config = parse_config_file(filename)
        for key in self._metadata.keys():
            self._metadata[key] = config[key]
        self.set_projection(**config.projection)
        self.set_extent(**config.extent)

    def set_projection(self, **kwargs):
        self._proj_dict = kwargs
        self._set_proj()

    def proj(self, longitude, latitude, **kwargs):
        projx, projy = self._proj(longitude, latitude, **kwargs)
        return projx, projy

    def grid_indices(self, longitude, latitude):
        """ Computes the grid indices the given lon/lat pairs would be sorted
        into (no clipping) """
        projx, projy = self.proj(longitude, latitude)
        extent = self.extent
        xi = np.floor((projx + extent.xsize/2.0)/extent.dx)
        yj = np.floor((projy + extent.ysize/2.0)/extent.dy)
        return xi, yj

    def get_grid_coordinates(self, mode="center"):
        """ Returns longitude/latitude points for each grid cell
        Note: mode keyword only for future use. center coordinates are
        returned by default """
        x0, y0 = self.extent.xoff, self.extent.yoff
        xsize, ysize = self.extent.xsize, self.extent.ysize
        numx, numy = self.extent.numx, self.extent.numy
        xmin, xmax = x0-(xsize/2.), x0+(xsize/2.)
        ymin, ymax = y0-ysize/2., y0+ysize/2.
        x = np.linspace(xmin, xmax, num=numx)
        y = np.linspace(ymin, ymax, num=numy)
        xx, yy = np.meshgrid(x, y)
        lon, lat = self.proj(xx, yy, inverse=True)
        return lon, lat

    def set_extent(self, **kwargs):
        self._extent_dict = kwargs

    def _set_proj(self):
        self._proj = Proj(**self._proj_dict)

    @property
    def hemisphere(self):
        return self._metadata["hemisphere"]

    @property
    def grid_id(self):
        return self._metadata["grid_id"]

    @property
    def grid_tag(self):
        return self._metadata["grid_tag"]

    @property
    def resolution_tag(self):
        return self._metadata["resolution_tag"]

    @property
    def extent(self):
        return TreeDict.fromdict(self._extent_dict, expand_nested=True)
