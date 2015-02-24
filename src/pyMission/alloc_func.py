from __future__ import division
import sys

import numpy as np

from openmdao.main.api import Component
from openmdao.main.datatypes.api import Array, Float, Bool


class Profit(Component):
    """ Computes the airline profit for the entire network """

    def __init__(self, num_routes, num_ac,
                 prc_pax, cost_fuel, cost_nf):
        super(Profit, self).__init__()

        self.prc_pax = prc_pax
        self.cost_fuel = cost_fuel
        self.cost_nf = cost_nf

        # Inputs
        self.add('pax_flt', Array(np.zeros((num_routes, num_ac)), iotype='in',
                                  desc = 'Passengers per flight'))
        self.add('flt_day', Array(np.zeros((num_routes, num_ac)), iotype='in',
                                  desc = 'Flights per day'))
        self.add('fuelburn', Array(np.zeros((num_routes, num_ac)), iotype='in',
                                  desc = 'Fuel burn'))

        # Outputs
        self.add('profit', Float(0.0, iotype='out', desc = 'Airline profit'))

    def execute(self):
        pax_flt, flt_day, fuelburn = self.pax_flt, self.flt_day, self.fuelburn * 1e5
        prc_pax, cost_fuel, cost_nf = self.prc_pax, self.cost_fuel, self.cost_nf

        self.profit = numpy.sum(prc_pax * pax_flt * flt_day) / 1e6 + \
                      numpy.sum((cost_fuel * fuelburn + cost_nf) * flt_day) / 1e6

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['pax_flt', 'flt_day', 'fuelburn']
        output_keys = ['profit']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        pax_flt, flt_day, fuelburn = self.pax_flt, self.flt_day, self.fuelburn * 1e5
        prc_pax, cost_fuel, cost_nf = self.prc_pax, self.cost_fuel, self.cost_nf

        if 'pax_flt' in arg:
            result['profit'] += numpy.sum(prc_pax * arg['pax_flt'] * flt_day) / 1e6
        if 'flt_day' in arg:
            result['profit'] += numpy.sum(prc_pax * pax_flt * arg['flt_day']) / 1e6 + \
                                numpy.sum((cost_fuel * fuelburn + cost_nf) * arg['flt_day']) / 1e6
        if 'fuelburn' in arg:
            result['profit'] += numpy.sum((cost_fuel * arg['fuelburn'] + cost_nf) * flt_day) / 1e6

    def apply_derivT(self, arg, result):
        pax_flt, flt_day, fuelburn = self.pax_flt, self.flt_day, self.fuelburn * 1e5
        prc_pax, cost_fuel, cost_nf = self.prc_pax, self.cost_fuel, self.cost_nf
        
        if 'pax_flt' in arg:
            result['pax_flt'] += prc_pax * arg['profit'] * flt_day / 1e6
        if 'flt_day' in arg:
            result['flt_day'] += prc_pax * pax_flt * arg['profit'] / 1e6 + \
                                (cost_fuel * fuelburn + cost_nf) * arg['profit'] / 1e6
        if 'fuelburn' in arg:
            result['fuelburn'] += (cost_fuel * arg['profit'] + cost_nf) * flt_day / 1e6
        

class PaxCon(Component):
    """ Computes constraint function for passenger demand """

    def __init__(self, num_routes, num_ac):
        super(PaxCon, self).__init__()

        # Inputs
        self.add('pax_flt', Array(np.zeros((num_routes, num_ac)), iotype='in',
                                  desc = 'Passengers per flight'))
        self.add('flt_day', Array(np.zeros((num_routes, num_ac)), iotype='in',
                                  desc = 'Flights per day'))

        # Outputs
        self.add('pax_con', Array(np.zeros(num_routes), iotype='out',
                                  desc = 'Demand constraint'))

    def execute(self):
        pax_flt, flt_day = self.pax_flt, self.flt_day

        self.pax_con = numpy.sum(pax_flt * flt_day, axis=1)

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['pax_flt', 'flt_day']
        output_keys = ['pax_con']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        pax_flt, flt_day = self.pax_flt, self.flt_day

        if 'pax_flt' in args:
            result['pax_con'] += numpy.sum(arg['pax_flt'] * flt_day, axis=1)
        if 'flt_day' in args:
            result['pax_con'] += numpy.sum(pax_flt * arg['flt_day'], axis=1)

    def apply_derivT(self, arg, result):
        pax_flt, flt_day = self.pax_flt, self.flt_day

        if 'pax_flt' in args:
            result['pax_flt'] += numpy.dot(numpy.diag(arg['pax_con']), flt_day)
        if 'flt_day' in args:
            result['flt_day'] += numpy.dot(numpy.diag(arg['pax_con']), pax_flt)


class AircraftCon(Component):
    """ Computes constraint function for aircraft availability """

    def __init__(self, num_routes, num_ac,
                 maintenance, turnaround):
        super(AircraftCon, self).__init__()

        self.maintenance, self.turnaround = maintenance, turnaround

        # Inputs
        self.add('flt_day', Array(np.zeros((num_routes, num_ac)), iotype='in',
                                  desc = 'Flights per day'))
        self.add('time', Array(np.zeros((num_routes, num_ac)), iotype='in',
                                     desc = 'Block time'))

        # Outputs
        self.add('ac_con', Array(np.zeros(num_ac), iotype='out',
                                 desc = 'Aircraft constraint'))

    def execute(self):
        flt_day, time = self.flt_day, self.time
        maintenance, turnaround = self.maintenance, self.turnaround

        self.ac_con = numpy.sum(flt_day * (time * maintenance + turnaround), axis=0)

    def list_deriv_vars(self):
        """ Return lists of inputs and outputs where we defined derivatives.
        """
        input_keys = ['flt_day', 'time']
        output_keys = ['ac_con']
        return input_keys, output_keys

    def provideJ(self):
        """ Calculate and save derivatives. (i.e., Jacobian) """
        pass

    def apply_deriv(self, arg, result):
        flt_day, time = self.flt_day, self.time
        maintenance, turnaround = self.maintenance, self.turnaround

        if 'flt_day' in args:
            result['ac_con'] += numpy.sum(arg['flt_day'] * (time * maintenance + turnaround), axis=0)
        if 'time' in args:
            result['ac_con'] += numpy.sum(flt_day * arg['time'] * maintenance, axis=0)

    def apply_derivT(self, arg, result):
        flt_day, time = self.flt_day, self.time
        maintenance, turnaround = self.maintenance, self.turnaround

        if 'flt_day' in args:
            result['flt_day'] += numpy.dot(time * maintenance + turnaround, numpy.diag(arg['ac_con']))
        if 'time' in args:
            result['time'] += numpy.dot(flt_day * maintenance, numpy.diag(arg['ac_con']))
