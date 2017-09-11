# -*- coding: utf-8 -*-

from pysimre.collection import DatasetOrbitCollection
from pysimre.dataset import CalValDataset
from pysimre.misc import ClassTemplate, parse_config_file

import glob
import os


class SimreRepository(ClassTemplate):
    """ Access container to local SIMBIE data repository """

    def __init__(self, local_path):
        """ Interface to local SIMRE dataset repository """
        super(SimreRepository, self).__init__(self.__class__.__name__)

        self._dataset_catalogues = {}
        self._calval_catalogue = None

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

    def get_orbit_collection(self, orbit_id):
        collection = DatasetOrbitCollection(orbit_id)
        orbit_dataset_list = [
                ctlg.filepath_info(orbit_id) for ctlg in self.catalogue_list
                if ctlg.has_orbit(orbit_id)]
        for dataset_id, filepath in orbit_dataset_list:
            collection.add_dataset(dataset_id, filepath)
        return collection

    def get_calval_dataset(self, orbit_id, source_id):
        """ Returns a calval dataset object """

        # check calval catalogue
        if not self.has_calval_orbit(orbit_id):
            msg = "calval product [%s] not in catalogue" % orbit_id
            self.error.add_error("invalid-orbit-id", msg)
            self.error.raise_on_error()

        ctlg_info = self._calval_catalogue.orbit_id_map[orbit_id]
        try:
            metadata = ctlg_info.source_id[source_id]
        except AttributeError:
            msg = "calval source [%s:%s] not in catalogue" % (
                    orbit_id, source_id)
            self.error.add_error("invalid-source-id", msg)
            self.error.raise_on_error()

        # Get filename and create data object
        filepath = self.get_calval_filepath(orbit_id)
        calval_dataset = CalValDataset(
                ctlg_info.pyclass,
                filepath,
                self.get_calval_dataset_id(orbit_id, source_id),
                orbit_id,
                metadata)

        return calval_dataset

    def get_calval_dataset_id(self, orbit_id, source_id):
        return orbit_id+"_"+source_id

    def get_figure_path(self, dataset_item):

        # Orbit Collection
        if isinstance(dataset_item, DatasetOrbitCollection):
            figure_path = os.path.join(
                    self.local_path,
                    "figures",
                    "calval_orbits",
                    dataset_item.orbit_id)
            # Create path (if needed)
            try:
                os.makedirs(figure_path)
            except WindowsError:
                pass
            return figure_path
        else:
            self.log.warning("Unknown dataset type: %s" % type(dataset_item))
            return None

    def has_calval_orbit(self, orbit_id):
        """ Tests if calval data for orbit id is known in the catalogue
        and target file exists """

        # Check for entry in calval config file
        def branches(t): return list(t.iterkeys(recursive=False,
                                     branch_mode='only'))

        ctlg = self._calval_catalogue
        has_orbit_entry = orbit_id in branches(ctlg.orbit_id_map)
        if not has_orbit_entry:
            return False

        # Check if file exists
        calval_filepath = self.get_calval_filepath(orbit_id)
        has_product_file = os.path.isfile(calval_filepath)
        if has_product_file:
            return True
        else:
            msg = "Entry for %s exist in calval config, file does not"
            msg = msg % orbit_id
            self.log.warning(msg)
            return False

    def get_calval_filepath(self, orbit_id):
        ctlg = self._calval_catalogue.orbit_id_map
        return os.path.join(self.local_calval_path,
                            ctlg[orbit_id].calval_filename)

    def _create_repo_catalogue(self):
        """ Scan the local repository for datasets (sat_products),
        ground truth (calval_products) and create a catalogue
        their content """

        # Get dataset id's in the local repository
        # XXX: Need to add a check for valid dataset ids
        result = os.listdir(self.local_satproduct_path)

        def is_dataset(p): return os.path.isdir(os.path.join(
                self.local_satproduct_path, p))

        dataset_ids = [p for p in result if is_dataset(p)]
        self._dataset_ids = dataset_ids

        # query directories for content
        for dataset_id in dataset_ids:
            path = os.path.join(self.local_satproduct_path, dataset_id)
            dataset_catalogue = SimreDatasetCatalogue(dataset_id, path)
            self._dataset_catalogues[dataset_id] = dataset_catalogue

        # Get a list of the calval product
        # XXX: At the moment only the contents of calval_product config file
        self._calval_catalogue = parse_config_file(self.calval_config_filepath)
        self._calval_catalogue.freeze()

    def get_grid_source_dataset(self, dataset_id, region_id, period_id):
        """ Return gridded data on the SIMRE default grid for a given
        dataset and period """
        ctlg = self.dataset_catalogues[dataset_id]
        filepath = ctlg.get_sourcegrid_filepath(region_id, period_id)

    @property
    def local_path(self):
        return str(self._local_path)

    @property
    def local_satproduct_path(self):
        return os.path.join(self.local_path, "sat_products")

    @property
    def local_calval_path(self):
        return os.path.join(self.local_path, "calval_products")

    @property
    def calval_config_filepath(self):
        calval_config_filepath = os.path.join(
                self.local_calval_path, "simre_calval_products_config.yaml")
        if not os.path.isfile(calval_config_filepath):
            msg = "Missing SIMRE calval config file. Expected: %s"
            msg = msg % str(calval_config_filepath)
            self.error.add_error("missing-calval-config", msg)
            self.error.raise_on_error()
        return calval_config_filepath

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

    @property
    def grid_dataset_ids(self):

        ctlg = self.dataset_catalogues

        def has_grid(dsi): return ctlg[dsi].has_grid_data

        return sorted([dsi for dsi in self.dataset_ids if has_grid(dsi)])


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

    def get_sourcegrid_filepath(self, region_id, period_id):
        rg_info = self.repo_config.dataset.region
        subfolders = [rg_info.subfolder]
        if rg_info.source_data.location == "inplace":
            subfolders.extend([region_id, period_id])
        else:
            subfolders.append(rg_info.location)
        file_path = os.path.join(self.path, *subfolders)
        stop

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

    @property
    def has_grid_data(self):
        return "region" in self.repo_config.dataset
