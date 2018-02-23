# -*- coding: utf-8 -*-
"""
Created on Fri Sep 08 11:37:27 2017

@author: Stefan
"""

from datetime import datetime
from dateutil.relativedelta import relativedelta

from scipy import stats as scistats

from itertools import product

import numpy as np

from pysimre.misc import ClassTemplate, pid2dt
from pysimre.dataset import OrbitThicknessDataset
from pysimre.output import GridReconciledNetCDF, OrbitReconciledNetCDF

# %% Orbit Collection classes

class OrbitCollection(ClassTemplate):

    def __init__(self, orbit_id):
        super(OrbitCollection, self).__init__(self.__class__.__name__)

        # Main identifier
        self._orbit_id = orbit_id

        # Datasets Satellite products
        self._datasets = {}
        self._calval_datasets = {}

        # Ensembles
        self._orbit_ensemble = None
        self._calval_ensemble = None
        self._ensemble_item_size_seconds = None

    def add_dataset(self, dataset_id, filepaths, metadata):
        """ Add an orbit thickness dataset to the collection """
        if type(filepaths) is list:
            multifile_dataset = OrbitThicknessDataset(dataset_id, filepaths[0], orbit=self.orbit_id,
                                                      metadata=metadata)
            for i in np.arange(1, len(filepaths)):
                dataset = OrbitThicknessDataset(dataset_id, filepaths[i], orbit=self.orbit_id,
                                                metadata=metadata)
                multifile_dataset.append(dataset)
            self._datasets[dataset_id] = multifile_dataset
        else:
            self._datasets[dataset_id] = OrbitThicknessDataset(dataset_id, filepaths, orbit=self.orbit_id, 
                                                               metadata=metadata)

    def get_dataset(self, dataset_id):
        """ Returns a OrbitThicknessDataset object for the given dataset_id.
        (None if dataset_id is not in the collection) """
        return self._datasets.get(dataset_id, None)

    def write_reconciled_netcdf(self, output_directory):
        nc = OrbitReconciledNetCDF(self.orbit_ensemble)
        nc.write(output_directory)
        

    def has_dataset(self, dataset_id):
        """ Return true of collection has the dataset `dataset_id` """
        return dataset_id in self.dataset_list

    def add_calval_dataset(self, calval_datset):
        self._calval_datasets[calval_datset.dataset_id] = calval_datset

    def has_calval_dataset(self, calval_dataset_id):
        return calval_dataset_id in self._calval_datasets

    def create_ensembles(self, ensemble_item_size_seconds):
        self._ensemble_item_size_seconds = ensemble_item_size_seconds
        self._create_orbit_ensemble(ensemble_item_size_seconds)
        self._create_calval_ensemble(ensemble_item_size_seconds)

    def _create_orbit_ensemble(self, ensemble_item_size_seconds):
        """ Computes time-based ensembles for all products """
        # Initialize the ensemble
        self._orbit_ensemble = OrbitDataEnsemble(
                self.orbit_id, self.time_range,
                ensemble_item_size_seconds)
        # Add ensemble members (datasets)
        for dataset in self:
            self._orbit_ensemble.add_member(dataset)
        self._orbit_ensemble.compute_reference_geolocations()

    def _create_calval_ensemble(self, ensemble_item_size_seconds):
        """Creates a cal/val ensemble for all cal/val datasets.
        Needs to be called after `create_orbit_ensemble` """

        # re-use information from satellite product ensemble
        ensemble_item_size_seconds = self._orbit_ensemble.member_size_seconds
        ref_time = self._orbit_ensemble.ref_time
        ref_lon = self._orbit_ensemble.longitude
        ref_lat = self._orbit_ensemble.latitude

        # Create the ensemble object
        self._calval_ensemble = OrbitDataEnsemble(
                self.orbit_id, self.time_range,
                ensemble_item_size_seconds,
                ref_time=ref_time, ref_lon=ref_lon, ref_lat=ref_lat)

        # Add cal/val datasets
        for calval_dataset_id in self.calval_dataset_ids:
            self._calval_ensemble.add_member(
                    self._calval_datasets[calval_dataset_id])

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
    def calval_dataset_ids(self):
        return sorted(self._calval_datasets.keys())

    @property
    def time_range(self):
        """ Returns the full time range of all datasets """
        time_ranges = np.array([dataset.time_range for dataset in self])
        return [np.amin(time_ranges[:, 0]), np.amax(time_ranges[:, 1])]

    @property
    def orbit_ensemble(self):
        return self._orbit_ensemble

    @property
    def calval_ensemble(self):
        return self._calval_ensemble

    @property
    def ensemble_item_size(self):
        return self._ensemble_item_size_seconds

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

    def __init__(self, orbit_id, time_range, member_size_seconds,
                 ref_time=None, ref_lon=None, ref_lat=None):
        super(OrbitDataEnsemble, self).__init__(self.__class__.__name__)
        self._members = {}
        self._orbit_id = orbit_id
        self._time_range = time_range
        self._member_size_seconds = member_size_seconds
        self._longitude = ref_lon
        self._latitude = ref_lat

        # The reference time can be known (e.g. from earlier datasets)
        if ref_time is None:
            self.create_reference_time()
        else:
            self._set_ref_time(*ref_time)

    def create_reference_time(self):
        """ Compute time range for each ensemble item """

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

        # Compute reference time for each item along the orbit
        for i in np.arange(n_ensembles):
            self._time_start[i] += relativedelta(seconds=second_deltas[i])
            self._time_stop[i] += relativedelta(seconds=second_deltas[i+1])
            center_second_delta = 0.5*(second_deltas[i+1]+second_deltas[i])
            self._time_center[i] += relativedelta(seconds=center_second_delta)

    def add_member(self, dataset):
        """ Add member to ensemble (must be orbit dataset) """
        ensemble_items = np.ndarray((self.n_ensemble_items), dtype=object)
        time_ranges = self.reftime_bounds
        for i, time_range in enumerate(time_ranges):
            parameters = dataset.get_ensemble_items(time_range)
            ensemble_items[i] = OrbitEnsembleItem(time_range, *parameters, metadata=dataset.metadata)
        self._members[dataset.dataset_id] = ensemble_items

    def get_dataset_metadata(self, dataset_id):
        metadata = [d[0].metadata for d in self.datasets if d[0].dataset_id == dataset_id]
        try: 
            return metadata[0]
        except IndexError:
            print metadata, dataset_id
            return None

    def get_member_mean(self, dataset_id):
        return np.array([m.mean for m in self._members[dataset_id]])

    def get_member_sdev(self, dataset_id):
        return np.array([m.sdev for m in self._members[dataset_id]])

    def get_member_points(self, dataset_id, mode="all"):
        """ Return the full resolution points for a dataset. Either all
        points (mode="all", default) or only for data points where all
        members of data (mode="colocated") """
        if mode == "all":
            indices = np.arange(self.n_ensemble_items)
        elif mode == "colocated":
            ncm = self.n_contributing_members
            indices = np.where(ncm == self.n_members)[0]
        else:
            msg = "Found %s, must be `all` or `colocated`" % str(mode)
            self.error.add_error("invalid-member-points-mode", msg)
            self.error.raise_on_error()
        points = []
        for index in indices:
            points.extend(self._members[dataset_id][index].point_list)
        return np.array(points)

    def get_ensemble_mean(self, n_members_min=2):
        """ Get the mean of all ensemble member mean values for each
        ensemble item interval. Must have more than n_members_min ensembles
        to compute an ensemble mean """
        ensemble_means = np.full((self.n_ensemble_items), np.nan)
        dataset_ids = self.dataset_ids
        indices = np.where(self.n_contributing_members >= n_members_min)[0]
        for i in indices:
            member_means = np.array([self._members[dataset_id][i].mean
                                    for dataset_id in dataset_ids])
            ensemble_means[i] = np.nanmean(member_means)
        return ensemble_means

    def get_ensemble_minmax(self, n_members_min=2):
        """ Get the min and max of all ensemble member mean values for each
        ensemble item interval. Must have more than n_members_min ensembles
        to compute a result """
        ensemble_min = np.full((self.n_ensemble_items), np.nan)
        ensemble_max = np.copy(ensemble_min)
        dataset_ids = self.dataset_ids
        indices = np.where(self.n_contributing_members >= n_members_min)[0]
        for i in indices:
            member_means = np.array([self._members[dataset_id][i].mean
                                    for dataset_id in dataset_ids])
            ensemble_min[i] = np.nanmin(member_means)
            ensemble_max[i] = np.nanmax(member_means)
        return ensemble_min, ensemble_max

    def compute_reference_geolocations(self):
        """ We cannot assume that the datasets are regulary spaced and the
        ensemble items have lat/lon values, or the lat/lon values might not
        fill the entire ensemble item. We therefore compute the center
        position from the max extent of all members for each item.
        Interpolation is used for items where no member has data """

        # Init
        self._longitude = np.full((self.n_ensemble_items), np.nan)
        self._latitude = np.full((self.n_ensemble_items), np.nan)

        # Loop over all items
        x = np.arange(self.n_ensemble_items)
        for i in x:

            lats = np.concatenate([self._members[name][i].lats
                                  for name in self.dataset_ids])
            lons = np.concatenate([self._members[name][i].lons
                                  for name in self.dataset_ids])

            # There might be the case where there are no valid lon/lat pairs for
            # a given time
            if len(lats) == 0:
                continue 

            # Get index of median of latitude value ()
            center_index = np.argsort(lats)[len(lats)//2]
            self._longitude[i] = lons[center_index]
            self._latitude[i] = lats[center_index]

        # Simple assumption that lat/lon will always be valid/invalid as pair
        valid = np.where(np.isfinite(self._longitude))[0]
        self._longitude = np.interp(x, x[valid], self._longitude[valid])
        self._latitude = np.interp(x, x[valid], self._latitude[valid])

    def _set_ref_time(self, t0, tc, t1):
        self._time_start = t0
        self._time_center = tc
        self._time_stop = t1

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
    def datasets(self): 
        return [self._members[did] for did in self.dataset_ids]

    @property
    def dataset_labels(self):
        return [id.replace("_", " ") for id in self.dataset_ids]

    @property
    def member_size_seconds(self):
        return self._member_size_seconds

    @property
    def n_ensemble_items(self):
        return len(self._time_start)

    @property
    def n_members(self):
        return len(self._members.keys())

    @property
    def n_contributing_members(self):
        """ Returns an array with the number of datasets contributing
        to ensemble item """
        n_contributing_members = np.full((self.n_ensemble_items), 0)
        for dataset_id in self.dataset_ids:
            has_data_int = np.array([
                    int(m.n_points > 0) for m in self._members[dataset_id]])
            n_contributing_members += has_data_int
        return n_contributing_members

    @property
    def reftime_bounds(self):
        t0, tc, t1 = self._time_start, self._time_center, self._time_stop
        index_list = np.arange(self.n_ensemble_items)
        return [(t0[i], tc[i], t1[i]) for i in index_list]

    @property
    def ref_time(self):
        return [self._time_start, self._time_center, self._time_stop]

    @property
    def time(self):
        __, tc, __ = self._time_start, self._time_center, self._time_stop
        index_list = np.arange(self.n_ensemble_items)
        return np.array([tc[i] for i in index_list])

    @property
    def longitude(self):
        return self._longitude

    @property
    def latitude(self):
        return self._latitude


class OrbitEnsembleItem(ClassTemplate):
    """ Container for data statistics of a single orbit ensemble item """

    def __init__(self, time_range, lons, lats, points, metadata=None):
        self.metadata = metadata
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
    def sdev(self):
        try:
            return np.nanstd(self._points)
        except ValueError:
            return np.nan

    @property
    def point_list(self):
        return list(self._points)

    @property
    def lats(self):
        return self._lats

    @property
    def lons(self):
        return self._lons

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

    @property
    def dataset_id(self):
        return self.metadata.id

# %% Grid Collection Classes


class GridCollection(ClassTemplate):

    def __init__(self, region_id, region_label=None):
        super(GridCollection, self).__init__(self)
        self._region_id = region_id
        self._region_label = region_label
        self._datasets = {}

    def add_dataset(self, dataset):
        """ Add a grid dataset """

        # Create a dictionary for a dataset (if not already existing)
        if dataset.dataset_id not in self.dataset_ids:
            self._datasets[dataset.dataset_id] = {}

        # add to dictionary in the struction dataset_id, period_id
        self._datasets[dataset.dataset_id][dataset.period_id] = dataset

    def get_dataset(self, dataset_id, period_id):
        try:
            return self._datasets[dataset_id][period_id]
        except KeyError:
            return None

    def get_ensemble(self, month):
        """ Returns an ensemble for the given month """

        # Get a list of applicable periods (e.g. all periods that include
        # defined month)
        ensemble_periods = self.get_ensemble_periods(month)

        # Create the ensemble object
        ensemble = GridDataEnsemble(self.region_id, ensemble_periods)

        # Add all applicable datasets to the ensemble
        ensemble_items = product(self.dataset_ids, ensemble_periods)
        for dataset_id, period_id in ensemble_items:
            try:
                ensemble.add_dataset(self._datasets[dataset_id][period_id])
            except KeyError:
                # Might not have a dataset for this period
                continue

        return ensemble

    def get_ensemble_periods(self, month):
        """ Returns a list of all periods in the collection
        for a defined month """

        month_str = "%02g" % month

        def is_month(p): return p.split("-")[1] == month_str

        return sorted([p for p in self.period_ids if is_month(p)])

    def get_period_subset(self, period_id):
        """
        Get a list of datasets for the given period id
            :param period_id: 
        """
        period_ensemble = GridRegionEnsemble(self.region_id, period_id, 
                                             region_label=self.region_label)
        for dataset_id in self.dataset_ids:
            try: 
                dataset = self._datasets[dataset_id][period_id]
            except ValueError:
                continue
            period_ensemble.add_dataset(dataset)
        return period_ensemble

    def write_reconciled_netcdf(self, reconciled_grid_dir):
        """
        Create a reconciled netcdf product for each period/region item
        """
        # Loop over all periods (one netcdf for each)
        for period_id in self.period_ids:
            # Set the id of the grid ensemble
            ensemble_id = "l3grid-sit-ensemble-%s-%s" % (self.region_id, period_id)
            period_ensemble = self.get_period_subset(period_id)
            ncfile = GridReconciledNetCDF(period_ensemble)
            ncfile.write(reconciled_grid_dir)

    @property
    def dataset_ids(self):
        """ Return a list of all dataset ids in collection """
        return sorted(self._datasets.keys())

    @property
    def period_ids(self):
        """ Return a list of all periods in collection. Note: Not all dataset
        might have a entry for all periods """
        period_ids = [self._datasets[did].keys() for did in self.dataset_ids]
        return np.unique(sorted(np.concatenate(period_ids)))

    @property
    def region_id(self):
        return str(self._region_id)

    @property
    def region_label(self):
        return str(self._region_label)

    @property
    def n_datasets(self):
        return len(self._datasets.keys())

    @property
    def n_periods(self):
        return len(self.period_ids)

    @property
    def thickness_range(self):
        min_vals, max_vals = [], []
        for dataset_id in self.dataset_ids:
            period_ids = sorted(self._datasets[dataset_id].keys())
            for period_id in period_ids:
                data = self.get_dataset(dataset_id, period_id)
                min_vals.append(np.nanmin(data.thickness))
                max_vals.append(np.nanmax(data.thickness))
        return np.nanmin(min_vals), np.nanmax(max_vals)


class GridRegionEnsemble(ClassTemplate):
    """
    An ensemble container for a defined region/period (ensemble items: different source datasets)
    """
    def __init__(self, region_id, period_id, region_label=None):
        super(GridRegionEnsemble, self).__init__(self.__class__.__name__)
        self._region_id = region_id
        self._region_label = region_label
        self._period_id = period_id
        self._datasets = []

    def add_dataset(self, dataset):
        """
        Add a dataset to the ensemble with safety checks
            :param dataset: 
        """
        # Sanity checks
        if dataset.region_id != self.region_id:
            msg = "region id do not match, skipping ... (ensemble: %s, dataset: %s)"
            self.log.warning(msg % (self.region_id, dataset.region_id))
            return 
        if dataset.period_id != self.period_id:
            msg = "period id do not match, skipping ... (ensemble: %s, dataset: %s)"
            self.log.warning(msg % (self.period_id, dataset.period_id))
            return
        if dataset.dataset_id in self.dataset_ids:
            msg = "duplicate dataset (%s), skipping ..."
            self.log.warning(msg % dataset.dataset_id)
            return
        self._datasets.append(dataset)
        if self.region_label is None:
            self._region_label = dataset.region_label

    def get_dataset_metadata(self, dataset_id):
        metadata = [d.metadata for d in self.datasets if d.dataset_id == dataset_id]
        try: 
            return metadata[0]
        except IndexError:
            return None

    @property
    def datasets(self):
        return list(self._datasets)

    @property
    def dataset_ids(self):
        return sorted([d.dataset_id for d in self.datasets])

    @property
    def region_id(self):
        return str(self._region_id)

    @property
    def region_label(self):
        return str(self._region_label)
        
    @property
    def period_id(self):
        return str(self._period_id)

    @property
    def sit_stack(self):
        return np.array([dst.thickness for dst in self.datasets])

    @property
    def n_datasets(self):
        return len(self.datasets)

    @property
    def longitude(self):
        return self.datasets[0].longitude

    @property
    def latitude(self):
        return self.datasets[0].latitude

    @property
    def mean_thickness(self):
        return np.nanmean(self.sit_stack, axis=0)

    @property
    def gmean_thickness(self):
        """ Computes the geometric mean """   
        sit_stack = self.sit_stack
        # scipy geometric mean requires masked arrays (it cannot handle NaN's)
        sit_stack = np.ma.array(sit_stack, mask=np.isnan(sit_stack))
        gmean = scistats.gmean(sit_stack, axis=0) 
        # convert back to norman array with NaN's
        gmean[gmean.mask] = np.nan
        return gmean

    @property
    def median_thickness(self):
        return np.nanmedian(self.sit_stack, axis=0)

    @property
    def min_thickness(self):
        return np.nanmin(self.sit_stack, axis=0)

    @property
    def max_thickness(self):
        return np.nanmax(self.sit_stack, axis=0)

    @property
    def thickness_stdev(self):
        return np.nanstd(self.sit_stack, axis=0)

    @property
    def n_points(self):
        sit_stack = self.sit_stack
        for i in np.arange(self.n_datasets):
            sit_stack[i, :, :] = np.isfinite(sit_stack[i, :, :]).astype(int)
        return np.sum(sit_stack, axis=0)


class GridDataEnsemble(ClassTemplate):

    def __init__(self, region_id, period_ids):
        """ Create a ensemble of thickness grid for a region and a series of
        periods """
        super(GridDataEnsemble, self).__init__(self.__class__.__name__)
        self._region_id = region_id
        self._period_ids = period_ids
        self._members = {}
        for period_id in self.period_ids:
            self._members[period_id] = {}

    def add_dataset(self, dataset):
        """ Add a datset (defined by dataset_id, period_id) to the ensemble
        with sanity checks """

        if not dataset.period_id in self.period_ids:
            msg = "Outside ensemble reference periods [%s, %s, %s], ignoring"
            msg = msg % (dataset.dataset_id, dataset.region_id,
                         dataset.period_id)
            self.log.warning(msg)
            return

        if not dataset.region_id == self.region_id:
            msg = "Outside ensemble reference periods [%s, %s, %s], ignoring"
            msg = msg % (dataset.dataset_id, dataset.region_id,
                         dataset.period_id)
            self.log.warning(msg)
            return

        # check passed -> add to ensemble
        ensemble_item = GridEnsembleItem(dataset)
        self._members[dataset.period_id][dataset.dataset_id] = ensemble_item

    def get_dataset_mean(self, dataset_id):
        """ Returns a date object and mean value for the a given dataset id
        in the ensemble time series """
        mean = []
        for period_id in self.period_ids:
            mean_val = self.get_dataset_period_mean(dataset_id, period_id)
            mean.append(mean_val)
        return mean

    def get_dataset_period_mean(self, dataset_id, period_id):
        try:
            ensemble_item = self._members[period_id][dataset_id]
            mean_val = ensemble_item.mean
        except KeyError:
            mean_val = np.nan
        return mean_val

    def get_ensemble_minmax(self):
        emin, emax = [], []
        dids = self.dataset_ids
        for period_id in self.period_ids:
            data_mean_vals = [self.get_dataset_period_mean(did, period_id)
                              for did in dids]
            emin.append(np.nanmin(data_mean_vals))
            emax.append(np.nanmax(data_mean_vals))
        return np.array(emin), np.array(emax)

    @property
    def ensemble_mean(self):
        ensemble_mean = []
        dids = self.dataset_ids
        for period_id in self.period_ids:
            data_mean_vals = [self.get_dataset_period_mean(did, period_id)
                              for did in dids]
            ensemble_mean.append(np.nanmean(data_mean_vals))
        return ensemble_mean

    @property
    def region_id(self):
        return str(self._region_id)

    @property
    def period_ids(self):
        return list(self._period_ids)

    @property
    def period_dts(self):
        return np.array(pid2dt(self.period_ids))

    @property
    def dataset_ids(self):
        dataset_ids = []
        for period_id in self.period_ids:
            dataset_ids.extend(sorted(self._members[period_id].keys()))
        return sorted(np.unique(dataset_ids))


class GridEnsembleItem(ClassTemplate):

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def dataset_id(self):
        return self._dataset.dataset_id

    @property
    def period_id(self):
        return self._dataset.period_id

    @property
    def region_id(self):
        return self._dataset.region_id

    @property
    def mean(self):
        return np.nanmean(self._dataset.thickness)