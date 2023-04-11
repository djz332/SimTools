import sys, os, copy
import subprocess

import numpy as np
import mdtraj as md

import sim
#from SimUtility_dev import *

#========== Sim Settings ==========#

units = sim.units.DimensionlessUnits

# Avogadro constant (1/mol).
N_A = 6.0221367e23
# Time onversion constant (unit/ps).
PS = sim.units.AtomicUnits.TimeScale

# === srel parameters and settings ===
sim.export.omm.platformName = 'OpenCL'
sim.export.omm.device = 0 # -1 is default, let openmm choose its own platform. otherwise is GPU device #
sim.srel.base.n_process = 8
print('NUM PROCESSES: {}'.format(sim.srel.base.n_process))

sim.export.omm.VVerbose = False
sim.srel.base.VVerbose = False
sim.srel.penalty.VVerbose = False
sim.srel.base.DEBUG = False
sim.srel.optimizetraj.PlotFmt = "png"
sim.srel.optimizetraj.DEBUGHESSIAN = True

#========== Simulation Parameters ==========#

# Mapping parameter
COM = True #True is center of mass, False is centroid

# Forcefield file (Sets initial forcefield parameters from an input file)
ForceFieldFile = 'final_matchA_ff.dat'

# Global forcefield parameters:
asmear = 0.310741
Kappa = 1.0/(4.0*asmear**2)
rcut = 5*asmear
Bmin = -100.

#electrostatic parameters
Electrostatics = True # turn on electrostatics and ewald summation 
NeutralSrel = True
Coef = 0.7152316677
BornA = asmear*np.sqrt(np.pi)
Shift = True

#thermostat:
Temp = 1.0

#barostat:
Pressure = 283.951857 # kT/nm^3, 0.0 for NVT
Axis = 2 # 2 for z-axis, etc.
Tension = 0.0 # 0.0 for MC-membrane barostat

#integrator:
dt = 0.1
LangevinGamma = 1.0

#molecular dynamics run:
MDEngine = 'openmm' # openmm or sim
StepsEquil = 500000
StepsProd = 500000
StepsStride = 5000

#========== System Definitions ==========#

#atom types: name, mass, charge
AtomTypes = [['W', 1.0, 0.0],
             ['C2', 1.0, 0.0],
             ['Na', 1.0, 1.0],
             ['Cl', 1.0, -1.0],
             ['SO4', 1.0, -1.0]]

#molecule types: name, list of CG atom types, #AA atoms per CG site
MolTypes = [['W', ['W'], [4]],
            ['Na', ['Na'], [1]],
            ['Cl', ['Cl'], [1]],
            ['C12', ['C2']*6, [7,6,6,6,6,7]],
            ['DS', ['SO4']+['C2']*6, [5,6,6,6,6,6,7]]]

#Bonds: molecule type, list of indices of bond pairs
Bonds = [['C12', [[0,1], [1,2], [2,3], [3,4], [4,5]]],
         ['DS', [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6]]]]

#systems: system name, molecule types, # of each moltype, box lengths (None to get from traj), neutralize
Systems = [['sys0', ['C12', 'DS', 'Na', 'W'], [179, 86, 86, 2956], [4.34, 4.34, 10.24]]]

#========== Forcefield Definitions ==========#

#harmonic bond: bond pair, Dist0 and FCont, fixed/unfixed
PBonds = [[['C2','C2'], [0.0, 16.493116675153434], [True,True]],
          [['C2','SO4'], [0.0, 63.064378588170115], [True,True]]]

#guassian potential: nonbonded pair, params(excl. vol., kappa), fixed during optimization 
PNonbonded = [[['W','W'], [0.5, Kappa], [True, True]],
              [['C2','C2'], [0.5, Kappa], [True, True]],
              [['C2','W'], [0.5, Kappa], [True, True]],
              [['Na','Na'], [0.5, Kappa], [True, True]],
              [['Cl','Cl'], [0.5, Kappa], [True, True]],
              [['Na','Cl'], [0.5, Kappa], [True, True]],
              [['Na','W'], [0.5, Kappa], [True, True]],
              [['Cl','W'], [0.5, Kappa], [True, True]],
              [['C2','SO4'], [0.5, Kappa], [False, True]],
              [['C2','Na'], [0.5, Kappa], [False, True]],
              [['C2','Cl'], [0.5, Kappa], [False, True]],
              [['SO4','SO4'], [0.5, Kappa], [False, True]],
              [['SO4','Na'], [0.5, Kappa], [False, True]],
              [['SO4','Cl'], [0.5, Kappa], [False, True]],
              [['SO4','W'], [0.5, Kappa], [False, True]],]

#smeared coulumb electrostatic potential: nonbonded pair, params(Bjerrum length, BornA), fixed during optimization
PElectrostatic = [[['Na','Na'], [Coef, BornA], [True, True]],
                  [['Cl','Cl'], [Coef, BornA], [True, True]],
                  [['Na','Cl'], [Coef, BornA], [True, True]],
                  [['SO4','Na'], [Coef, BornA], [True, True]],
                  [['SO4','Cl'], [Coef, BornA], [True, True]],
                  [['SO4','SO4'], [Coef, BornA], [True, True]]]


#external potentials: type(guassian, sine), atom types, potential parameters
#PExternal = [['sinusoid', ['Na'], [0.0, 1, 2, 0.0]],
#             ['sinusoid', ['Br'], [0.0, 1, 2, 0.0]]]

#========== Measures ==========#

#Ree^2 measure: name, system, molecule type, molecule sites
Ree2 = [['Ree2', 'sys0', 'DS', [1,6]]]
StageCoefsRee2 = [10.0, 100.0, 1000.0]

#interfacial area measure: name, system, axis
Area = [['Az', 'sys0', 2]]
StageCoefsA = [10.0, 100.0, 1000.0]

#========== Relative Entropy Arguments ==========#

#trajectories: name, trajectory file, topology file, format
Trajectories = [['traj0', '../output_post.dcd', '../equilibrated.pdb', 'dcd']]

#optimizers: optimizer name, systems, trajectories, file prefix
Optimizers = [['Opt0', 'sys0', 'traj0', 'SDS_interface']]

#Penalties: optimizer name, measure name, measure target, stage coefficients
Penalties = [['Opt0', 'Az', 18.82698]]
#             ['Opt0', 'Ree2', 0.95]]

#========== Run Protocal ==========#

#Run: optimizer, #0=both 1=srel only 2=bias only 3=CGMD, stage coefficients, weights (multitraj only)
Runs = [[['Opt0'], 0, StageCoefsA, [1., 1.]],
        [['Opt0'], 3, StageCoefsA, [1., 1.]]]

