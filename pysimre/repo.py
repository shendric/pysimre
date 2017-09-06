# -*- coding: utf-8 -*-

from pysimre.misc import ClassTemplate, parse_config_file

import glob
import os


class SimreRepository(ClassTemplate):
    """ Access container to local SIMBIE data repository """

    def __init__(self, local_path):
        """ Interface to local SIMRE dataset repository """
        super(SimreRepository, self).__init__(self.__class__.__name__)

        self._dataset_catalogues = {}

        # Validity check of local path
        if not os.path.isdir(local_path):
            msg = "Invalid path: {0}".format(local_path)
            self.error.add_error("invalid-path", msg)
            self.error.raise_on_error()
        self._local_path = local_path

        # Scan the local directory for content
        self._create_repo_catalogue()

    def get_dataset_catalogue(self, dataset_id):
        return self.dataset_catalogues.get(dataset_id, None)

    def _create_repo_catalogue(self):
        """ Scan the local repository for data sets and create a catalogue
        their content """

        # Get dataset id's in the local repository
        # XXX: Need to add a check for valid dataset ids
        result = os.listdir(self.local_path)
        dataset_ids = [p for p in result
                       if os.path.isdir(os.path.join(self.local_path, p))]
        self._dataset_ids = dataset_ids

        # query directories for content
        for dataset_id in dataset_ids:
            path = os.path.join(self.local_path, dataset_id)
            dataset_catalogue = SimreDatasetCatalogue(dataset_id, path)
            self._dataset_catalogues[dataset_id] = dataset_catalogue

    @property
    def local_path(self):
        return str(self._local_path)

    @property
    def dataset_ids(self):
        """ dictionary of dataset object """
        return dict(self._dataset_ids)

    @property
    def n_datasets(self):
        """ Number of datasets in the repository """
        return len(self.datasets.keys())

    @property
    def dataset_catalogues(self):
        return dict(self._dataset_catalogues)


class SimreDatasetCatalogue(ClassTemplate):
    """ A catalogue of all files in the SIMRE repository for a given
    dataset """

    def __init__(self, dataset_id, path):
        super(SimreDatasetCatalogue, self).__init__(self.__class__.__name__)
        # Store input parameters
        self._dataset_id = dataset_id
        self._path = path
        # Read the configuration files
        self.repo_config = parse_config_file(self.repo_config_file)
        self.repo_config.freeze()
        # List all orbit files
        self.query_orbit_files()

    def query_orbit_files(self):
        """ Simple list of all files in the dataset repository """
        search_str = os.path.join(
                self.orbit_data_path,
                self.repo_config.dataset.orbit.search_pattern)
        self._orbit_files = sorted(glob.glob(search_str))

    @property
    def dataset_id(self):
        return str(self._dataset_id)

    @property
    def path(self):
        return str(self._path)

    @property
    def orbit_data_path(self):
        repo_subfolder = self.repo_config.dataset.orbit.subfolder
        return os.path.join(self.path, repo_subfolder)

    @property
    def repo_config_file(self):
        """ Return path to configuration file in the local repository.
        Raise an error if it does not exist """
        config_file_path = os.path.join(self.path, "simre_dataset_config.yaml")
        if not os.path.isfile(config_file_path):
            msg = "Missing SIMRE dataset config file. Expected: %s"
            msg = msg % str(config_file_path)
            self.error.add_error("missing-dataset-config", msg)
            self.error.raise_on_error()
        return config_file_path
