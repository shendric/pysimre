# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 15:19:39 2017

@author: shendric
"""

import warnings

import numpy as np

import pyproj

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap


class OrbitParameterGraph(object):

    def __init__(self, dataset, parameter_name, block=True, ylim="auto"):
        """ Generate a simple matplotlib graph of a given parameter """

        # Some sanity checks
        if dataset.n_records == 0:
            warnings.warn("No data")
            return

        if parameter_name not in dataset.parameter_list:
            warnings.warn("Parameter [%s] does not exist" % parameter_name)
            return

        # Get value for abcissa
        if dataset.has_timestamp:
            x = dataset.timestamp
        else:
            x = np.arange(dataset.n_records)

        # Get parameter
        y = getattr(dataset, parameter_name)

        label = "$parameter$:%s,  $source$:%s,  $orbit$:%s" % (
                parameter_name, dataset.source_id, dataset.orbit)

        # Make the plot
        gs = gridspec.GridSpec(1, 5)
        plt.figure(label, figsize=(12, 6))

        plt.subplots_adjust(bottom=0.075, top=0.95, left=0.1, right=0.9)

        ax0 = plt.subplot(gs[0:-1])
        # Histogram axes
        ax1 = plt.subplot(gs[-1], sharey=ax0)

        ax0.plot(x, y, lw=0.5, alpha=0.5)
        ax0.scatter(date2num(x), y, marker=".")
        ax0.set_title(label, loc="left")
        if ylim != "auto":
            plt.ylim(ylim)

        valid = np.where(np.isfinite(self.sea_ice_thickness))
        hist, bin_edges = np.histogram(
                self.sea_ice_thickness[valid], bins=50, density=True)
        bin_width = bin_edges[1]-bin_edges[0]
        bin_center = bin_edges[0:-1] + 0.5*bin_width
        ax1.barh(bin_center, hist, height=bin_width)
        ax1.yaxis.set_ticks_position('right')

        plt.show(block=block)


class OrbitParameterMap(object):

    def __init__(self, dataset, block=True, basemap_args=None):
        """ Generate a simple map with orbit location """

        has_thickness = np.where(np.isfinite(dataset.sea_ice_thickness))[0]

        grid = {
            'coarse': {
                'color': '1.0',
                'dashes': [],
                'linewidth': 0.25,
                'fontsize': 8,
                'zorder': 50},
            'fine': {
                'color': '1.0',
                'dashes': [],
                'linewidth': 0.1,
                'fontsize': 8,
                'zorder': 50}
                }

        label = "$source$:%s,  $orbit$:%s" % (
                dataset.source_id, dataset.orbit)
        if basemap_args is None:
            basemap_args = get_basemap_args_from_positions(self, scale=1.5)
        plt.figure(figsize=(6, 6))
        m = Basemap(**basemap_args)
        m.drawmapboundary(linewidth=0.5, fill_color='0.9', zorder=200)
        m.drawcoastlines(linewidth=0.25, color="0.5")
        m.fillcontinents(color="0.8", lake_color="0.8", zorder=100)
        for type in ['coarse']:
            parallels, keyw = self._get_parallels(grid, type)
            m.drawparallels(parallels, **keyw)
            meridians, keyw = self._get_meridians(grid, type)
            m.drawmeridians(meridians, **keyw)
        px, py = m(self.longitude, self.latitude)
        m.scatter(px[has_thickness], py[has_thickness], marker=".", zorder=200)
        plt.title(label, loc="left")
        plt.show(block=block)

        return basemap_args

    def _get_parallels(self, grid, type):
        latmax = 88
        latstep = 4
        latlabels = [0, 0, 0, 0]
        pad = 90.0 - latmax
        if type == 'coarse':
            parallels = np.arange(-90+pad, 91-pad, latstep)
        elif type == 'fine':
            parallels = np.arange(-90+pad, 91-pad, latstep/2.0)
        else:
            raise ValueError('type must be ''coarse'' or ''fine''')
        keywords = {
            'labels': latlabels,
            'color': grid[type]["color"],
            'dashes': grid[type]["dashes"],
            'linewidth': grid[type]["linewidth"],
            'fontsize': grid[type]["fontsize"],
            'latmax': latmax,
            'zorder': grid[type]["zorder"]}
        return parallels, keywords

    def _get_meridians(self, grid, type):
        latmax = 88
        lonstep = 30
        lonlabels = [0, 0, 0, 0]
        if type == 'coarse':
            meridians = np.arange(0, 360, lonstep)
        elif type == 'fine':
            meridians = np.arange(0, 360, lonstep/2.0)
        else:
            raise ValueError('type must be ''coarse'' or ''fine''')
        keywords = {
            'labels': lonlabels,
            'color': grid[type]["color"],
            'dashes': grid[type]["dashes"],
            'linewidth': grid[type]["linewidth"],
            'fontsize': grid[type]["fontsize"],
            'latmax': latmax,
            'zorder': grid[type]["zorder"]}
        return meridians, keywords


def get_basemap_args_from_positions(pos, aspect=1, scale=1.1, res='h'):
    """ Get basemap parameters that display the given positions
    in a sterographic map with a given aspect (map width to map height) and
    a scale (1 = no padding) """

    # lon_0 = np.mean([np.nanmin(pos.longitude), np.nanmax(pos.longitude)])
    # lat_0 = np.mean([np.nanmin(pos.latitude), np.nanmax(pos.latitude)])
    lat_0 = np.median(pos.latitude)
    lon_0 = np.median(pos.longitude)
    p = pyproj.Proj(proj='stere', lon_0=lon_0, lat_0=lat_0, ellps='WGS84')
    x, y = p(pos.longitude, pos.latitude)
    width_r = scale*(np.nanmax(x)-np.nanmin(x))
    height_r = scale*(np.nanmax(y)-np.nanmin(y))
    maxval = np.amax([width_r, height_r])
    # Get the edges
    width = maxval
    height = maxval
    if aspect > 1:
        width *= aspect
    if aspect < 1:
        height *= aspect

    basemap_kwargs = {'projection': 'stere',
                      'width': width,
                      'height': height,
                      'lon_0': lon_0,
                      'lat_0': lat_0,
                      'lat_ts': lat_0,
                      'resolution': res}

    return basemap_kwargs
