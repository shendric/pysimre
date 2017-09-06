# -*- coding: utf-8 -*-
"""
Created on Wed May 10 22:16:36 2017

@author: Stefan
"""
import warnings


class DirectOrbitComparison(object):

    def __init__(self):
        self._products = []
        self._orbit = None

    def clip_to_latbox(self, lat_limits, direction):
        """
        Clip all orbits to a latitude box either for ascending
        or descending orbit direction
        """
        if self.n_products == 0:
            warnings.warn("No products to clip")
            return
        for product in self._products:
            product.clip_to_latbox(lat_limits, direction)

    def add_product(self, product):

        # TODO: Add object type check

        # Do not allow adding products from different orbits
        # - First product sets the reference orbit number
        # - all following products must have matching orbit numbers
        if self._orbit is None:
            self._orbit = product.orbit
        else:
            assert self._orbit == product.orbit

        self._products.append(product)

    @property
    def n_products(self):
        return len(self._products)