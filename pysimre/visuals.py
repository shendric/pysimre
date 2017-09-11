# -*- coding: utf-8 -*-
"""
Created on Wed Sep 06 15:19:39 2017

@author: shendric
"""

import warnings

from pysimre.misc import ClassTemplate

import numpy as np
import os

import pyproj

import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap
from matplotlib.mlab import poly_between
import matplotlib.patheffects as path_effects
from matplotlib.collections import LineCollection
from matplotlib import markers
from matplotlib.path import Path

DATASET_COLOR = {"awi": "#00ace5",
                 "ucl": "#000000",
                 "ccicdr": "#00dc6e",
                 "nasa_jpl": "#ff0e0e"}

DATASET_MARKER = {"awi": "D",
                  "ucl": "s",
                  "ccicdr": "o",
                  "nasa_jpl": "H"}


class OrbitCollectionHists(ClassTemplate):
    """ A figure summarizin the thickness histograms for all/colocated
    ensemble points """

    bg_color_fig = "0.96"
    bg_color_ax = "0.96"

    def __init__(self, orbit_collection, output_path):

        super(OrbitCollectionHists, self).__init__(self.__class__.__name__)

        # Save input parameter
        self._oc = orbit_collection
        self._output_path = output_path

        plt.ioff()
        self._create_figure()
        self._populate_subplots()
        self._add_metadata()
        self._save_to_file()

    def _create_figure(self):

        # Create figure
        self.fig = plt.figure(figsize=(12, 5))
        panels = (1, 3)

        self.ax_hist_all = plt.subplot2grid(panels, (0, 0))
        self.ax_hist_all.set_zorder(100)
        self.ax_hist_col = plt.subplot2grid(panels, (0, 1))
        self.ax_hist_col.set_zorder(100)
        self.ax_map = plt.subplot2grid(panels, (0, 2))
        self.ax_map.set_zorder(100)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.80, bottom=0.15,
                            wspace=0.35, hspace=0.25)

        # Highlight graphs
        bbox_ens = self.ax_hist_col.get_position()
        bbox_map = self.ax_map.get_position()
        x1 = 0.5*(bbox_ens.x1+bbox_map.x0)
        bg = self.fig.add_axes([0, 0, x1, 1])
        bg.xaxis.set_visible(False)
        bg.yaxis.set_visible(False)
        bg.set_zorder(5)
        bg.patch.set_color(self.bg_color_fig)
        for target in ["left", "right", "bottom", "top"]:
            bg.spines[target].set_visible(False)

    def _populate_subplots(self):

        # Plot the satellite product ensembles
        orbit_ensemble = self._oc.orbit_ensemble

        # Create a graphical representation of the sat & cal/val ensembles
        OrbitEnsembleHist(self.ax_hist_all, self._oc, mode="all", means=True)
        set_axes_style(self.fig, self.ax_hist_all, bg_color=self.bg_color_ax,
                       remove_yaxis=True)
        self.ax_hist_all.set_xlabel("Sea Ice Thickness (m)")
        self.ax_hist_all.set_title("All Datapoints")

        # Create a graphical representation of the sat & cal/val ensembles
        OrbitEnsembleHist(self.ax_hist_col, self._oc, mode="colocated",
                          means=True)
        set_axes_style(self.fig, self.ax_hist_col, bg_color=self.bg_color_ax,
                       remove_yaxis=True)
        self.ax_hist_col.set_xlabel("Sea Ice Thickness (m)")
        self.ax_hist_col.set_title("Colocated Datapoints")

        # Overview Map
        cmap = plt.get_cmap("RdYlGn", lut=orbit_ensemble.n_members)
        alongtrack_args = dict(norm=plt.Normalize(0, orbit_ensemble.n_members),
                               cmap=cmap)
        lon, lat = orbit_ensemble.longitude, orbit_ensemble.latitude
        colorbar = dict(ticks=np.arange(0, orbit_ensemble.n_members+1),
                        label="Number Ensemble Member")
        OrbitParameterMap(self.ax_map, lon, lat,
                          zval=orbit_ensemble.n_contributing_members,
                          alongtrack_args=alongtrack_args,
                          colorbar=colorbar)
        self.ax_map.set_title(self._oc.orbit_id)

    def _add_metadata(self):

        orbit_ensemble = self._oc.orbit_ensemble

        # get right limit
        x0 = self.ax_hist_all.get_position().x0
        x1 = self.ax_hist_col.get_position().x1

        x_pad = 0.1*(x1-x0)
        label_x0 = x0+x_pad
        label_x1 = x1-x_pad

        label_x = np.linspace(label_x0, label_x1, orbit_ensemble.n_members)
        for i, dataset_id in enumerate(orbit_ensemble.dataset_ids):
            color = DATASET_COLOR[dataset_id]
            plt.annotate(dataset_id.replace("_", " "), (label_x[i], 0.91),
                         xycoords="figure fraction",
                         ha="center", color=color, fontsize=16)

    def _save_to_file(self):
        plt.savefig(self.output_filename, dpi=300)

    @property
    def output_filename(self):
        filename = "OC_%03g_%s_hists.png"
        filename = filename % (self._oc.ensemble_item_size, self._oc.orbit_id)
        return os.path.join(self._output_path, filename)


class OrbitCollectionGraph(ClassTemplate):
    """ A figure summarizing the orbit collection """

    bg_color_fig = "0.96"
    bg_color_ax = "0.96"

    def __init__(self, orbit_collection, output_path):

        super(OrbitCollectionGraph, self).__init__(self.__class__.__name__)

        # Save input parameter
        self._oc = orbit_collection
        self._output_path = output_path

        plt.ioff()
        self._create_figure()
        self._populate_subplots()
        self._add_metadata()
        self._save_to_file()

    def _create_figure(self):

        # Create figure
        self.fig = plt.figure(figsize=(12, 10))

        self.ax_ens = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        self.ax_ens.set_zorder(100)
        self.ax_ensr = plt.subplot2grid((3, 3), (1, 0), colspan=2)
        self.ax_ensr.set_zorder(100)
        self.ax_map = plt.subplot2grid((3, 3), (0, 2))
        self.ax_ensh = plt.subplot2grid((3, 3), (2, 0))
        self.ax_ensh.set_zorder(100)
        self.ax_ensscat = plt.subplot2grid((3, 3), (2, 1))
        self.ax_ensscat.set_zorder(100)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1,
                            wspace=0.35, hspace=0.25)

        # Highlight graphs
        bbox_ens = self.ax_ens.get_position()
        bbox_map = self.ax_map.get_position()
        x1 = 0.5*(bbox_ens.x1+bbox_map.x0)
        bg = self.fig.add_axes([0, 0, x1, 1])
        bg.xaxis.set_visible(False)
        bg.yaxis.set_visible(False)
        bg.set_zorder(5)
        bg.patch.set_color(self.bg_color_fig)
        for target in ["left", "right", "bottom", "top"]:
            bg.spines[target].set_visible(False)

    def _populate_subplots(self):

        # Plot the satellite product ensembles
        orbit_ensemble = self._oc.orbit_ensemble

        # Create a graphical representation of the sat & cal/val ensembles
        OrbitEnsembleGraph(self.ax_ens, self._oc, calval=True)
        set_axes_style(self.fig, self.ax_ens, bg_color=self.bg_color_ax)
        self.ax_ens.set_ylabel("Sea Ice Thickness (m)")

        # Create a graphical representation of the sat & cal/val residuals
        OrbitEnsembleResidualGraph(self.ax_ensr, self._oc, calval=True)
        set_axes_style(self.fig, self.ax_ensr, bg_color=self.bg_color_ax)
        self.ax_ensr.set_ylabel("Thickness Ensemble Anomaly (m)")

        # Create a graphical representation of the sat & cal/val residuals
        OrbitEnsembleResidualHist(self.ax_ensh, self._oc)
        set_axes_style(self.fig, self.ax_ensh, bg_color=self.bg_color_ax)
        self.ax_ensh.set_xlabel("Thickness Ensemble Anomaly (m)")
        self.ax_ensh.set_ylabel("Frequency")

        # Create a graphical representation of the sat & cal/val residuals
        OrbitEnsembleScatterGraph(self.ax_ensscat, self._oc)
        set_axes_style(self.fig, self.ax_ensscat, bg_color=self.bg_color_ax)
        self.ax_ensscat.set_xlabel("Ensemble Mean Thickness (m)")
        self.ax_ensscat.set_ylabel("Ensemble Spread (m)")

        # Overview Map
        lon, lat = orbit_ensemble.longitude, orbit_ensemble.latitude
        OrbitParameterMap(self.ax_map, lon, lat)

    def _add_metadata(self):

        orbit_ensemble = self._oc.orbit_ensemble

        # get right limit
        y0 = self.ax_map.get_position().y0
        x1 = self.ax_map.get_position().x1

        # Orbit ID
        y = y0 - 0.05
        plt.annotate("Orbit ID", (x1, y), xycoords="figure fraction",
                     ha="right")
        y -= 0.03
        plt.annotate(self._oc.orbit_id, (x1, y), xycoords="figure fraction",
                     ha="right", fontsize=14)

        y -= 0.1
        plt.annotate("Datasets", (x1, y), xycoords="figure fraction",
                     ha="right")
        y -= 0.05
        # Label all datasets
        for dataset_id in orbit_ensemble.dataset_ids:
            color = DATASET_COLOR[dataset_id]
            plt.annotate(dataset_id, (x1, y), xycoords="figure fraction",
                         ha="right", color=color, fontsize=20)
            y -= 0.04

    def _save_to_file(self):
        plt.savefig(self.output_filename, dpi=300)

    @property
    def output_filename(self):
        filename = "OC_%03g_%s.png"
        filename = filename % (self._oc.ensemble_item_size, self._oc.orbit_id)
        return os.path.join(self._output_path, filename)


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

        valid = np.where(np.isfinite(dataset.sea_ice_thickness))
        hist, bin_edges = np.histogram(
                dataset.sea_ice_thickness[valid], bins=50, density=True)
        bin_width = bin_edges[1]-bin_edges[0]
        bin_center = bin_edges[0:-1] + 0.5*bin_width
        ax1.barh(bin_center, hist, height=bin_width)
        ax1.yaxis.set_ticks_position('right')

        plt.show(block=block)


class OrbitEnsembleGraph(object):

    def __init__(self, ax, orbit_collection, ensemble_mean=True,
                 calval=True):

        # Prepare Input
        orbit_ensemble = orbit_collection.orbit_ensemble
        orbit_ensemble_mean = orbit_ensemble.get_ensemble_mean()
        calval_ensemble = orbit_collection.calval_ensemble

        # Plot all datasets
        for dataset_id in orbit_ensemble.dataset_ids:

            color = DATASET_COLOR[dataset_id]
            marker = DATASET_MARKER[dataset_id]

            # Get statistics for each dataset
            dataset_mean = orbit_ensemble.get_member_mean(dataset_id)

            # Plot the mean as lines and symbols
#            ax.plot(orbit_ensemble.time[is_valid], dataset_mean[is_valid],
#                    color=color, **self.datasetl_props)
            ax.scatter(orbit_ensemble.time, dataset_mean, color=color,
                       marker=marker, **self.datasets_props)

        # Plot ensemble means
        if ensemble_mean:
            ax.plot(orbit_ensemble.time, orbit_ensemble_mean,
                    label="Ensemble Mean Thickness",
                    **self.ensemble_mean_props)

            # Fill in between ensemble min/max
            emin, emax = orbit_ensemble.get_ensemble_minmax()
            is_valid = np.where(np.isfinite(dataset_mean))

            # Plot standard deviation envelope in the background
            is_valid = np.where(np.isfinite(emin))
            xs, ys = poly_between(orbit_ensemble.time[is_valid],
                                  emin[is_valid], emax[is_valid])
            ax.fill(xs, ys, **self.sdev_fill_props)

        # Plot the calval ensembles
        if calval:
            shadow = [path_effects.SimpleLineShadow(offset=(1, -1)),
                      path_effects.Normal()]
            for calval_id in calval_ensemble.dataset_ids:
                dataset_mean = calval_ensemble.get_member_mean(calval_id)
                ax.scatter(calval_ensemble.time, dataset_mean,
                           path_effects=shadow, **self.calval_props)

        ax.set_xlim(orbit_collection.time_range)
        leg = ax.legend(**self.legend_props)
        leg.set_zorder(300)

    @property
    def sdev_fill_props(self):
        return dict(alpha=0.1, edgecolor="none", zorder=200)

    @property
    def datasetl_props(self):
        return dict(lw=0.5, alpha=0.25, zorder=210)

    @property
    def datasets_props(self):
        return dict(s=12, alpha=0.5, edgecolors="none", zorder=211)

    @property
    def ensemble_mean_props(self):
        return dict(color="black", lw=2, alpha=0.5, zorder=220)

    @property
    def calval_props(self):
        return dict(lw=2, facecolors="none", edgecolors="#EE00EE", zorder=230)

    @property
    def legend_props(self):
        return dict(frameon=False)


class OrbitEnsembleResidualGraph(object):

    def __init__(self, ax, orbit_collection, calval=True):

        # Prepare Input
        orbit_ensemble = orbit_collection.orbit_ensemble
        orbit_ensemble_mean = orbit_ensemble.get_ensemble_mean()
        calval_ensemble = orbit_collection.calval_ensemble

        # Plot all datasets
        for dataset_id in orbit_ensemble.dataset_ids:

            color = DATASET_COLOR[dataset_id]
            marker = DATASET_MARKER[dataset_id]

            # Get statistics for each dataset
            dataset_mean = orbit_ensemble.get_member_mean(dataset_id)

            ax.scatter(orbit_ensemble.time, dataset_mean-orbit_ensemble_mean,
                       color=color,  marker=marker, **self.datasets_props)

        # Plot the calval ensembles
        if calval:
            shadow = [path_effects.SimpleLineShadow(offset=(1, -1)),
                      path_effects.Normal()]
            for calval_id in calval_ensemble.dataset_ids:
                dataset_mean = calval_ensemble.get_member_mean(calval_id)
                ax.scatter(calval_ensemble.time,
                           dataset_mean-orbit_ensemble_mean,
                           path_effects=shadow, **self.calval_props)

        ax.axhline(0, **self.zero_line_props)
        ax.set_ylim(-2, 2)
        ax.set_xlim(orbit_collection.time_range)

    @property
    def sdev_fill_props(self):
        return dict(alpha=0.1, edgecolor="none", zorder=200)

    @property
    def datasetl_props(self):
        return dict(lw=0.5, alpha=0.25, zorder=210)

    @property
    def datasets_props(self):
        return dict(s=12, alpha=0.5, edgecolors="none", zorder=211)

    @property
    def ensemble_mean_props(self):
        return dict(color="black", lw=2, alpha=0.5, zorder=220)

    @property
    def zero_line_props(self):
        return dict(color="0.1", lw=1, alpha=0.75, zorder=100)

    @property
    def calval_props(self):
        return dict(lw=2, facecolors="none", edgecolors="#EE00EE", zorder=230)


class OrbitEnsembleResidualHist(object):

    def __init__(self, ax, orbit_collection):

        orbit_ensemble = orbit_collection.orbit_ensemble
        orbit_ensemble_mean = orbit_ensemble.get_ensemble_mean()

        bins = np.arange(-2.1, 2.1, 0.2)

        # Plot all datasets
        for dataset_id in orbit_ensemble.dataset_ids:

            color = DATASET_COLOR[dataset_id]

            # Get statistics for each dataset
            dataset_mean = orbit_ensemble.get_member_mean(dataset_id)
            ensemble_residual = dataset_mean-orbit_ensemble_mean
            is_valid = np.where(np.isfinite(ensemble_residual))[0]
            ax.hist(ensemble_residual[is_valid], bins, facecolor=color,
                    normed=True, **self.hist_props1)
            ax.hist(ensemble_residual[is_valid], bins, color=color,
                    histtype='step', normed=True, **self.hist_props2)

        ax.set_xlim(-2, 2)
        # ax.set_ylim(0, 1.1)
        # ax.axvline(0, )

    @property
    def hist_props1(self):
        return dict(alpha=0.1, zorder=200, edgecolor="none")

    @property
    def hist_props2(self):
        return dict(alpha=0.5, zorder=200, lw=0.5)


class OrbitEnsembleHist(object):

    def __init__(self, ax, orbit_collection, mode="all", means=True):

        orbit_ensemble = orbit_collection.orbit_ensemble

        bins = np.arange(-1.1, 10.1, 0.2)

        max_freq_value = []

        # Plot all datasets
        for dataset_id in orbit_ensemble.dataset_ids:

            color = DATASET_COLOR[dataset_id]

            # Get statistics for each dataset
            points = orbit_ensemble.get_member_points(dataset_id, mode=mode)
            is_valid = np.where(np.isfinite(points))[0]
            n, bins, patches = ax.hist(points[is_valid], bins, facecolor=color,
                    normed=True, **self.hist_props1)
            ax.hist(points[is_valid], bins, color=color,
                    histtype='step', normed=True, **self.hist_props2)

            max_freq_value.append(np.amax(n))

        # Typical thickness range
        ax.set_xlim(-1, 8)

        # Frequency range
        max_freq = np.nanmax(max_freq_value)
        minval = 0
        maxval = 1.2*max_freq
        ax.set_ylim(minval, maxval)

        if means:
            # Plot all datasets
            for id in orbit_ensemble.dataset_ids:
                color = DATASET_COLOR[id]
                # Get statistics for each dataset
                points = orbit_ensemble.get_member_points(id, mode=mode)
                meanval = np.nanmean(points)
                ax.scatter(meanval, 1.01*max_freq,
                           marker=align_marker('v', valign='bottom'),
                           s=240, lw=0.75, color=color,
                           facecolors="none", alpha=0.75)

    @property
    def hist_props1(self):
        return dict(alpha=0.1, zorder=200, edgecolor="none")

    @property
    def hist_props2(self):
        return dict(alpha=0.5, zorder=200, lw=0.5)


class OrbitEnsembleScatterGraph(object):

    def __init__(self, ax, orbit_collection):

        # Prepare Input
        orbit_ensemble = orbit_collection.orbit_ensemble
        orbit_ensemble_mean = orbit_ensemble.get_ensemble_mean()

        # Get min max
        emin, emax = orbit_ensemble.get_ensemble_minmax()
        ensemble_spread = emax-emin

        ax.scatter(orbit_ensemble_mean, ensemble_spread,
                   **self.scatter_props)

        ax.set_xlim(0, 5)
        ax.set_ylim(-0.1, 2)

    @property
    def scatter_props(self):
        return dict(marker="s", s=20, color="0.1", edgecolor="none",
                    zorder=200)


class OrbitParameterMap(object):

    def __init__(self, ax, lons, lats, basemap_args=None,
                 zval=None, alongtrack_args=None, colorbar=None):
        """ Generate a simple map with orbit location """

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

#        label = "$source$:%s,  $orbit$:%s" % (
#                dataset.source_id, dataset.orbit)
        if basemap_args is None:
            basemap_args = get_basemap_args_from_positions(
                    lons, lats, scale=1.5)
        m = Basemap(ax=ax, **basemap_args)
        m.drawmapboundary(linewidth=0.1, fill_color='#CCE5FF', zorder=200)
        m.drawcoastlines(linewidth=0.25, color="0.5")
        m.fillcontinents(color="1.0", lake_color="0.96", zorder=100)
        for type in ['coarse']:
            parallels, keyw = self._get_parallels(grid, type)
            m.drawparallels(parallels, **keyw)
            meridians, keyw = self._get_meridians(grid, type)
            m.drawmeridians(meridians, **keyw)
        px, py = m(lons, lats)

        if zval is not None:
            points = np.array([px, py]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, **alongtrack_args)
            lc.set_array(zval)
            lc.set_linewidth(3)
            lc.set_zorder(200)
            ax.add_collection(lc)
        else:
            m.scatter(px, py, marker=".", zorder=200)


        if zval is not None and colorbar is not None:
            cb = plt.colorbar(lc, ax=ax, shrink=0.7, aspect=15,
                              drawedges=False, orientation="horizontal",
                              **colorbar)
            # cb.solids.set_visible(False)


        #plt.title(label, loc="left")

        # return basemap_args

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


def set_axes_style(f, ax, bg_color=None, spines_to_remove=["top", "right"],
                   axes_pad=-0.03, remove_yaxis=False):

    if bg_color is not None:
        ax.patch.set_color(bg_color)

    if remove_yaxis:
        ax.yaxis.set_visible(False)
        spines_to_remove.extend(["left", "right"])
        spines_to_remove = np.unique(spines_to_remove)

    # Remove missor axis
    for spine in spines_to_remove:
        ax.spines[spine].set_visible(False)



#    cl = plt.getp(ax, 'xmajorticklabels')
#    plt.setp(cl, **self.monthfontprops)



    # y-axis
    fig_size = f.get_size_inches()
    ax_bbox = ax.get_position()
    ax_asp = (ax.bbox.y1-ax.bbox.y0)/(ax.bbox.x1-ax.bbox.x0)
    fig_asp = fig_size[1]/fig_size[0]
    #stop
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines["bottom"].set_position(("axes", axes_pad))

    ax.yaxis.set_tick_params(direction='out')
    ax.yaxis.set_ticks_position('left')
    ax.spines["left"].set_position(("axes", axes_pad*ax_asp))

#    cl = plt.getp(ax, 'ymajorticklabels')
#    plt.setp(cl, **self.parlabelfontprops)


def get_basemap_args_from_positions(longitude, latitude,
                                    aspect=1, scale=1.1, res='h'):
    """ Get basemap parameters that display the given positions
    in a sterographic map with a given aspect (map width to map height) and
    a scale (1 = no padding) """

    # lon_0 = np.mean([np.nanmin(pos.longitude), np.nanmax(pos.longitude)])
    # lat_0 = np.mean([np.nanmin(pos.latitude), np.nanmax(pos.latitude)])
    lat_0 = np.median(latitude)
    lon_0 = np.median(longitude)
    p = pyproj.Proj(proj='stere', lon_0=lon_0, lat_0=lat_0, ellps='WGS84')
    x, y = p(longitude, latitude)
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


def align_marker(marker, halign='center', valign='middle',):
    """
    create markers with specified alignment.

    Parameters
    ----------

    marker : a valid marker specification.
      See mpl.markers

    halign : string, float {'left', 'center', 'right'}
      Specifies the horizontal alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'center',
      -1 is 'right', 1 is 'left').

    valign : string, float {'top', 'middle', 'bottom'}
      Specifies the vertical alignment of the marker. *float* values
      specify the alignment in units of the markersize/2 (0 is 'middle',
      -1 is 'top', 1 is 'bottom').

    Returns
    -------

    marker_array : numpy.ndarray
      A Nx2 array that specifies the marker path relative to the
      plot target point at (0, 0).

    Notes
    -----
    The mark_array can be passed directly to ax.plot and ax.scatter, e.g.::

        ax.plot(1, 1, marker=align_marker('>', 'left'))

    """

    if isinstance(halign, (str, unicode)):
        halign = {'right': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'left': 1.,
                  }[halign]

    if isinstance(valign, (str, unicode)):
        valign = {'top': -1.,
                  'middle': 0.,
                  'center': 0.,
                  'bottom': 1.,
                  }[valign]

    # Define the base marker
    bm = markers.MarkerStyle(marker)

    # Get the marker path and apply the marker transform to get the
    # actual marker vertices (they should all be in a unit-square
    # centered at (0, 0))
    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    # Shift the marker vertices for the specified alignment.
    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return Path(m_arr, bm.get_path().codes)
