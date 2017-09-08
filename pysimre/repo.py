# -*- coding: utf-8 -*-

from pysimre.collection import DatasetOrbitCollection
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

    def get_orbit_collection(self, orbit_id, ensemble_item_size_seconds=1):
        collection = DatasetOrbitCollection(orbit_id)
        orbit_dataset_list = [
                ctlg.filepath_info(orbit_id) for ctlg in self.catalogue_list
                if ctlg.has_orbit(orbit_id)]
        for dataset_id, filepath in orbit_dataset_list:
            collection.add_dataset(dataset_id, filepath)
        collection.create_orbit_ensemble(ensemble_item_size_seconds)
        return collection

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
        return self._dataset_ids

    @property
    def n_datasets(self):
        """ Number of datasets in the repository """
        return len(self.datasets.keys())

    @property
    def dataset_catalogues(self):
        return dict(self._dataset_catalogues)

    @property
    def catalogue_list(self):
        return [self.dataset_catalogues[dsi] for dsi in self.dataset_ids]


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
        self.query_orbit_filemap()

    def query_orbit_files(self):
        """ Simple list of all files in the dataset repository """
        search_str = os.path.join(
                self.orbit_data_path,
                self.repo_config.dataset.orbit.search_pattern)
        self._orbit_files = sorted(glob.glob(search_str))

    def query_orbit_filemap(self):
        """ Get a dictionary that provides list of filenames for
        orbit ids (need to be defined in `simre_dataset_config.yaml`)"""
        repr_cfg = self.repo_config.dataset.orbit
        try:
            self._orbit_filemap = repr_cfg.orbit_id_filemap
        except AttributeError:
            self.log.warning("orbit_id_filemap missing [%s]" % self.dataset_id)
            self._orbit_filemap = {}

    def has_orbit(self, orbit_id):
        """ Returns true or false, depending on whether a certain
        orbit_id is known in `simre_dataset_config.yaml` """
        return orbit_id in self.orbit_filemap_list

    def filepath_info(self, orbit_id):
        return (self.dataset_id,
                os.path.join(self.orbit_data_path,
                             self.orbit_filemap.get(orbit_id, None)))

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
    def orbit_filemap_list(self):
        """ Returns a simple list of all orbit files """
        return sorted(self.orbit_filemap.keys())

    @property
    def orbit_filemap(self):
        """ Returns a simple list of all orbit files """
        return self._orbit_filemap

    @property
    def orbit_files(self):
        """ Returns a dictionary of all orbit file maps """
        return dict(self._orbit_files)

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
