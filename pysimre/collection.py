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
