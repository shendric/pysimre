# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 11:37:27 2017

@author: Stefan
"""

from pysimre.misc import ClassTemplate
from pysimre.dataset import OrbitThicknessDataset

from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np


class DatasetOrbitCollection(ClassTemplate):

    def __init__(self, orbit_id):
        super(DatasetOrbitCollection, self).__init__(self.__class__.__name__)
        self._orbit_id = orbit_id
        self._datasets = {}
        self._ensemble = None

    def add_dataset(self, dataset_id, filepath):
        """ Add an orbit thickness dataset to the collection """
        self._datasets[dataset_id] = OrbitThicknessDataset(
                dataset_id, filepath, orbit=self.orbit_id)

    def get_dataset(self, dataset_id):
        """ Returns a OrbitThicknessDataset object for the given dataset_id.
        (None if dataset_id is not in the collection) """
        return self._datasets.get(dataset_id, None)

    def has_dataset(self, dataset_id):
        """ Return true of collection has the dataset `dataset_id` """
        return dataset_id in self.dataset_list

    def create_orbit_ensemble(self, ensemble_item_size_seconds):
        """ Computes time-based ensembles for all products """
        # Initialize the ensemble
        self._ensemble = OrbitDataEnsemble(
                self.orbit_id, self.time_range,
                ensemble_item_size_seconds)
        # Add ensemble members (datasets)
        for dataset in self:
            self._ensemble.add_member(dataset)
        self._ensemble.compute_reference_geolocations()

    @property
    def n_datasets(self):
        return len(self._datasets.keys())

    @property
    def dataset_list(self):
        return sorted(self._datasets.keys())

    @property
    def orbit_id(self):
        return str(self._orbit_id)

    @property
    def time_range(self):
        """ Returns the full time range of all datasets """
        time_ranges = np.array([dataset.time_range for dataset in self])
        return [np.amin(time_ranges[:, 0]), np.amax(time_ranges[:, 1])]

    @property
    def ensemble(self):
        return self._ensemble

    def __repr__(self):
        msg = "SIMRE orbit dataset collection:\n"
        msg += "       Orbit id : %s\n" % self.orbit_id
        msg += "   Datasets (%s) : %s" % (str(self.n_datasets),
                                          str(self.dataset_list))
        return msg

    def __getitem__(self, index):
        dataset_id = self.dataset_list[index]
        result = self._datasets[dataset_id]
        return result


class OrbitDataEnsemble(ClassTemplate):

    def __init__(self, orbit_id, time_range, member_size_seconds):
        super(OrbitDataEnsemble, self).__init__(self.__class__.__name__)
        self._members = {}
        self._orbit_id = orbit_id
        self._time_range = time_range
        self._member_size_seconds = member_size_seconds
        self._longitude = None
        self._latitude = None
        self.create_reference_time()

    def create_reference_time(self):
        """ Compute time range for each ensemble member """
        # prepare input
        tr = self.time_range
        duration = tr[1]-tr[0]
        total_seconds = duration.total_seconds()
        size = self.member_size_seconds
        second_deltas = np.arange(0, total_seconds+size, size)
        epoch = datetime(tr[0].year, tr[0].month, tr[0].day, tr[0].hour,
                         tr[0].minute, tr[0].second)
        # compute start, center, stop time per ensemble
        n_ensembles = len(second_deltas)-1
        self._time_start = np.full((n_ensembles), epoch)
        self._time_stop = np.full((n_ensembles), epoch)
        self._time_center = np.full((n_ensembles), epoch)
        for i in np.arange(n_ensembles):
            self._time_start[i] += relativedelta(seconds=second_deltas[i])
            self._time_stop[i] += relativedelta(seconds=second_deltas[i+1])
            center_second_delta = 0.5*(second_deltas[i+1]+second_deltas[i])
            self._time_center[i] += relativedelta(seconds=center_second_delta)

    def add_member(self, dataset):
        """ Add member to ensemble (must be orbit dataset) """
        ensemble_items = np.ndarray((self.n_emsemble_items), dtype=object)
        time_ranges = self.reftime_bounds
        for i, time_range in enumerate(time_ranges):
            parameters = dataset.get_ensemble_items(time_range)
            ensemble_items[i] = OrbitEnsembleItem(time_range, *parameters)
        self._members[dataset.dataset_id] = ensemble_items

    def get_member_mean(self, dataset_id):
        return np.array([m.mean for m in self._members[dataset_id]])

    def compute_reference_geolocations(self):
        """ We cannot assume that the datasets are regulary spaced and the
        ensemble items have lat/lon values, or the lat/lon values might not
        fill the entire ensemble item. We therefore compute the center
        position from the max extent of all members for each item.
        Interpolation is used for items where no member has data """

        # Init
        self._longitude = np.full((self.n_emsemble_items), np.nan)
        self._latitude = np.full((self.n_emsemble_items), np.nan)

        # Loop over all items
        x = np.arange(self.n_emsemble_items)
        for i in x:
            lat_min = np.nanmin([self._members[name][i].lat_min
                                 for name in self.dataset_ids])
            lat_max = np.nanmax([self._members[name][i].lat_max
                                 for name in self.dataset_ids])
            lon_min = np.nanmin([self._members[name][i].lon_min
                                 for name in self.dataset_ids])
            lon_max = np.nanmax([self._members[name][i].lon_max
                                 for name in self.dataset_ids])
            self._longitude[i] = 0.5*(lon_max-lon_min)+lon_min
            self._latitude[i] = 0.5*(lat_max-lat_min)+lat_min

        # Simple assumption that lat/lon will always be valid/invalid as pair
        valid = np.where(np.isfinite(self._longitude))[0]
        self._longitude = np.interp(x, x[valid], self._longitude[valid])
        self._latitude = np.interp(x, x[valid], self._latitude[valid])

    @property
    def orbit_id(self):
        return str(self._orbit_id)

    @property
    def time_range(self):
        return list(self._time_range)

    @property
    def dataset_ids(self):
        return sorted(self._members.keys())

    @property
    def member_size_seconds(self):
        return self._member_size_seconds

    @property
    def n_emsemble_items(self):
        return len(self._time_start)

    @property
    def n_members(self):
        return len(self._members.keys())

    @property
    def reftime_bounds(self):
        t0, tc, t1 = self._time_start, self._time_center, self._time_stop
        index_list = np.arange(self.n_emsemble_items)
        return [(t0[i], tc[i], t1[i]) for i in index_list]

    @property
    def time(self):
        __, tc, __ = self._time_start, self._time_center, self._time_stop
        index_list = np.arange(self.n_emsemble_items)
        return np.array([tc[i] for i in index_list])

    @property
    def longitude(self):
        return self._longitude

    @property
    def latitude(self):
        return self._latitude


class OrbitEnsembleItem(ClassTemplate):
    """ Container for data statistics of a single orbit ensemble item """

    def __init__(self, time_range, lons, lats, points):
        self._time_range = time_range
        self._lons = lons
        self._lats = lats
        self._points = points

    @property
    def n_points(self):
        return len(self._points)

    @property
    def mean(self):
        try:
            return np.nanmean(self._points)
        except ValueError:
            return np.nan

    @property
    def min(self):
        try:
            return np.nanmin(self._points)
        except ValueError:
            return np.nan

    @property
    def max(self):
        try:
            return np.nanmax(self._points)
        except ValueError:
            return np.nan

    @property
    def point_list(self):
        return list(self._points)

    @property
    def lat_min(self):
        try:
            return np.nanmin(self._lats)
        except ValueError:
            return np.nan

    @property
    def lat_max(self):
        try:
            return np.nanmax(self._lats)
        except ValueError:
            return np.nan

    @property
    def lon_min(self):
        try:
            return np.nanmin(self._lons)
        except ValueError:
            return np.nan

    @property
    def lon_max(self):
        try:
            return np.nanmax(self._lons)
        except ValueError:
            return np.nan
