# -*- coding: utf-8 -*-
import glob
import os
import re

import numpy as np
from itertools import product

from pysimre import REGION_DEF_FILENAME

from pysimre.collection import OrbitCollection, GridCollection
from pysimre.dataset import CalValDataset, GridSourceData, RegionGrid
from pysimre.misc import ClassTemplate, parse_config_file


class SimreRepository(ClassTemplate):
    """ Access container to local SIMBIE data repository """

    def __init__(self, local_path):
        """ Interface to local SIMRE dataset repository """
        super(SimreRepository, self).__init__(self.__class__.__name__)

        self._dataset_catalogues = {}
        self._calval_catalogue = None
        self._region_def = None

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
        collection = OrbitCollection(orbit_id)
        orbit_dataset_list = [
            ctlg.filepath_info(orbit_id) for ctlg in self.catalogue_list
                if ctlg.has_orbit(orbit_id)]
        for dataset_id, filepath in orbit_dataset_list:
            collection.add_dataset(dataset_id, filepath)
        return collection

    def get_grid_collection(self, region_id):
        """ Returns a collection object for all gridded datasets
        (files of type SIMRE_$dataset_id$_$region_id$_$period_id%.nc) """

        # Init the collection object
        grid_collection = GridCollection(region_id)

        # Loop over all datasets and add a data set for each data period
        for dataset_id in self.dataset_ids:

            # Get the catalogue for the dataset and retrieve all
            # SIMRE grid files
            ctlg = self.dataset_catalogues[dataset_id]
            simre_netcdfs = ctlg.get_simre_grid_netcdfs(region_id=region_id)

            # Add the region grid
            # (which is aware of dataset_id, region_id & period_id)
            for simre_netcdf in simre_netcdfs:
                region_grid = RegionGrid(self._local_path, region_id)
                region_grid.from_netcdf(simre_netcdf)
                grid_collection.add_dataset(region_grid)

        return grid_collection

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
        if isinstance(dataset_item, OrbitCollection):
            figure_path = os.path.join(
                    self.local_path,
                    "figures",
                    "calval_orbits",
                    dataset_item.orbit_id)

        elif isinstance(dataset_item, GridCollection):
            figure_path = os.path.join(
                    self.local_path,
                    "figures",
                    "test_region",
                    dataset_item.region_id)

        else:
            self.log.warning("Unknown dataset type: %s" % type(dataset_item))
            return None

        # Create path (if needed)
        try:
            os.makedirs(figure_path)
        except WindowsError:
            pass
        return figure_path

    def get_simre_grid_path(self, region_data):
        """ Get the full path for a SIMRE region grid """
        ctlg = self.dataset_catalogues[region_data.dataset_id]
        ids = [region_data.dataset_id, region_data.region_id,
               region_data.period_id]
        return ctlg.get_simre_grid_filepath(*ids)

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

        # Get region ids from region definition file
        self._region_def = parse_config_file(self.region_def_config_filepath)
        self._region_def.freeze()

    def get_grid_source_dataset(self, dataset_id, region_id, period_id):
        """ Return gridded data on the SIMRE default grid for a given
        dataset and period """
        ctlg = self.dataset_catalogues[dataset_id]
        filepath = ctlg.get_sourcegrid_filepath(region_id, period_id)
        pyclass = ctlg.sourcegrid_pyclass
        ids = [dataset_id, region_id, period_id]

        # Check if file has been found
        if filepath is None:
            self.log.warning("No source file in data catalogues")
            return None

        # Check if file(s) exists
        # NOTE:
        # 1. If multiple specific input file period exist, the file check
        #    counts as failed if one source files is missing
        # 2. If the filepath is a search string (*), no meaningful file check
        #    can be carried out here
        if "*" not in filepath:
            if isinstance(filepath, list):
                subfiles_exist = [os.path.isfile(f) for f in filepath]
                file_exists = False not in subfiles_exist
            else:
                file_exists = os.path.isfile(filepath)

            if not file_exists:
                msg = "Missing input files for %s.%s.%s" % tuple(ids)
                self.log.warning(msg)
                return None

        # Get the source data
        self.log.info("Retrieving grid source data [%s.%s.%s]" % tuple(ids))

        # Get the source data on the SIMRE grid
        source_data_grid = GridSourceData(
                pyclass, filepath, self._local_path, *ids)

        # Check for errors while retrieving and gridding data
        if not source_data_grid.error.status:
            return source_data_grid
        return None

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
    def region_def_config_filepath(self):
        region_def_config_filepath = os.path.join(self.local_path, REGION_DEF_FILENAME)
        if not os.path.isfile(region_def_config_filepath):
            msg = "Missing SIMRE region definition config file. Expected: %s"
            msg = msg % str(region_def_config_filepath)
            self.error.add_error("missing-calval-config", msg)
            self.error.raise_on_error()
        return region_def_config_filepath

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

    @property
    def grid_region_ids(self):
        region_ids = self._region_def.iterkeys(branch_mode="only")
        return sorted(region_ids)

    @property
    def grid_period_ids(self):
        period_ids = []
        ctlg = self.dataset_catalogues
        for dataset_id in self.dataset_ids:
            period_ids.extend(ctlg[dataset_id].grid_period_ids)
        return sorted(np.unique(period_ids))

    @property
    def ensemble_pool(self):
        return product(self.grid_region_ids, self.grid_period_ids)

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
        """ Return the path to local source grid files """
        rg_info = self.repo_config.dataset.region
        try:
            subfolders = [rg_info.subfolder]
        except AttributeError:
            msg = "expected %s in %s" % (
                    "root.dataset.region.subfolder",
                    self.repo_config_file)
            self.error.add_error("simre-config-error", msg)
            self.error.raise_on_error()
        if rg_info.source_data.location == "inplace":
            subfolders.extend([region_id, period_id])
        else:
            try:
                subfolders.append(rg_info.source_data.location)
            except AttributeError:
                msg = "expected %s in %s" % (
                        "root.dataset.region.source_data.location",
                        self.repo_config_file)
                self.error.add_error("simre-config-error", msg)
                self.error.raise_on_error()

        # Directory of data
        file_path = os.path.join(self.path, *subfolders)

        # Get link to source file(s)
        try:
            period_key = "period_"+period_id.replace("-", "_")
            files = rg_info.source_data.file_map[period_key]
            if isinstance(files, list):
                return [os.path.join(file_path, f) for f in files]
            return os.path.join(file_path, files)
        except KeyError:
            return None

        return file_path

    def get_simre_grid_filepath(self, dataset_id, region_id, period_id):
        rg_info = self.repo_config.dataset.region
        subfolders = [rg_info.subfolder, region_id]
        directory = os.path.join(self.path, *subfolders)
        filename = "SIMRE-%s-%s-%s.nc" % (dataset_id, region_id, period_id)
        return os.path.join(directory, filename)

    def get_simre_grid_netcdfs(self, region_id=None):
        """ Return a list of SIMRE grid netcdf files """

        # Config data
        rg_info = self.repo_config.dataset.region

        # Support function

        def is_netcdf(f): return re.search("SIMRE-(\S)*.nc", f)

        # Lookup directory
        directory = os.path.join(self.path, rg_info.subfolder)
        if region_id is not None:
            directory = os.path.join(directory, region_id)

        # Check for all SIMRE files in lookup directory
        # (search independent of lookup directory subfolders)
        simre_grid_netcdfs = []

        for root, dirs, files in os.walk(directory):
            netcdfs = [os.path.join(root, f) for f in files if is_netcdf(f)]
            simre_grid_netcdfs.extend(sorted(netcdfs))

        return simre_grid_netcdfs

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
    def grid_period_ids(self):
        """ Returns a list of periods. This is taken from the config file of the repository
        (dataset.region.source_data.file_map) """
        if not self.has_grid_data:
            return []
        grid_file_map = self.repo_config.dataset.region.source_data.file_map
        period_keys = sorted(grid_file_map.keys())
        period_ids = [key[-7:].replace('_', '-') for key in period_keys]
        return period_ids

    @property
    def has_grid_data(self):
        return "region" in self.repo_config.dataset

    @property
    def sourcegrid_pyclass(self):
        return self.repo_config.dataset.region.source_data.pyclass
