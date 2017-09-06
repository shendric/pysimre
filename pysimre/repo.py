# -*- coding: utf-8 -*-

import os


class SimbieRepository(object):
    """ Access container to local SIMBIE data repository """

    def __init__(self, local_path):
        """ """

        self._datasets = {}

        # Validity check of local path
        if not os.path.isdir(local_path):
            raise IOError("Invalid path: {0}".format(local_path))
        self._local_path = local_path

        # Scan the local directory for content
        self._create_repo_catalogue()

    def _create_repo_catalogue(self):
        """ Scan the local repository for data sets and create a catalogue
        their content """

    @property
    def local_path(self):
        return self._local_path

    @property
    def datasets(self):
        """ dictionary of dataset object """
        return dict(self._datasets)

    @property
    def n_datasets(self):
        """ Number of datasets in the repository """
        return len(self.datasets.keys())


class SimbieDataset(object):

    def __init__(self, dataset_id):
        self._id = dataset_id