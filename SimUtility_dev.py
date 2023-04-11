import os, sys, copy
import subprocess
import time

import argparse as ap
import numpy as np
import mdtraj as md

import sim

class Parameters:

     def __init__(self, COM, ForceFieldFile, asmear, rcut, Bmin, Electrostatics, NeutralSrel, Coef, BornA, Shift, Temp, Pressure, Axis, Tension, dt, LangevinGamma, MDEngine, StepsEquil, StepsProd, StepsStride):

        self._COM = COM
        self._ForceFieldFile = ForceFieldFile
        self._asmear = asmear
        self._rcut = rcut
        self._Bmin = Bmin
        self._Electrostatics = Electrostatics
        self._NeutralSrel = NeutralSrel
        self._Coef = Coef
        self._BornA = BornA
        self._Shift = Shift
        self._Temp = Temp
        self._Pressure = Pressure
        self._Axis = Axis
        self._Tension = Tension
        self._dt = dt
        self._LangevinGamma = LangevinGamma
        self._MDEngine = MDEngine
        self._StepsEquil = StepsEquil
        self._StepsProd = StepsProd
        self._StepsStride = StepsStride

def SaveHessian(Opt):

    H = Opt.DDObj
    #H = [k[k != 0.0] for k in H]
    #H = np.asarray(H)
    #n = int(np.sqrt(len(H)))
    #H.reshape(n,n)
    Use = ~Opt.ConstrMask
    H = H[np.ix_(Use, Use)]
    print('\nHessian for {}:\n{}'.format(Opt.FilePrefix, H))
    np.savetxt('{}_Hessian.dat'.format(Opt.FilePrefix), H)
    return

def CreateSystem(params, Systems, AtomTypes, MolTypes, Bonds):

    print('\n========== Defining System ==========')
    atom_dict = {}
    mol_dict = {}
    Sys_dict = {}

    for entry in AtomTypes:
        atom_dict[entry[0]] = sim.chem.AtomType(entry[0], Mass = entry[1], Charge = entry[2])
    print('Defined atom types:')
    print(atom_dict)

    for entry in MolTypes:                  
        atomtypes = [atom_dict[aname] for aname in entry[1]]
        mol_dict[entry[0]] = sim.chem.MolType(entry[0], atomtypes)
    
    for entry in Bonds:
        for bond in entry[1]:
            mol_dict[entry[0]].Bond(bond[0],bond[1])
    print('Defined molecule types:')
    print(mol_dict)

    World = sim.chem.World([mol_dict[entry[0]] for entry in MolTypes], Dim = 3, Units = sim.units.DimensionlessUnits)
    
    for Sys_entry in Systems:
        Sys = sim.system.System(World, Name = Sys_entry[0])
        print('\nCreating system: {}'.format(Sys.Name))
        for molname, nmol in zip(Sys_entry[1], Sys_entry[2]):
            print('Adding {} {} to system'.format(nmol, molname))
            for n in range(nmol):
                Sys+= mol_dict[molname].New()
        if Sys_entry[3] != None: Sys.BoxL = Sys_entry[3]
        Sys_dict[Sys_entry[0]] = Sys

        # ElecSys: system with Ewald on that is used to run MD, used to speed up when only optimizing non-charged interactions for speed up
        if  params._Electrostatics and params._NeutralSrel: 
            #ElecSys = copy.deepcopy(Sys) 
            #ElecSys.name = 'ElecSys_{}'.format(Sys.name)
            ElecSys = sim.system.System(World, Name = 'ElecSys_{}'.format(Sys.Name))
            print('\nCreating system: {}'.format(ElecSys.Name))
            for molname, nmol in zip(Sys_entry[1], Sys_entry[2]):
                print('Adding {} {} to system'.format(nmol, molname))
                for n in range(nmol):
                 ElecSys += mol_dict[molname].New()
            if Sys_entry[3] != None: ElecSys.BoxL = Sys_entry[3]
            Sys_dict[ElecSys.Name] = ElecSys

    for Sys in Sys_dict.values():
        Sys.TempSet = params._Temp
        Sys.PresSet = params._Pressure
        Sys.BarostatOptions['tension'] = params._Tension
        Sys.PresAx = params._Axis
        Sys.Int.Method = Sys.Int.Methods.VVIntegrate
        Sys.Int.Method.Thermostat = Sys.Int.Method.ThermostatLangevin
        Sys.Int.Method.TimeStep = params._dt
        Sys.Int.Method.LangevinGamma = params._LangevinGamma
        
    return atom_dict, mol_dict, Sys_dict

def SetupPotentials(params, PBonds, PNonbonded, PElectrostatic, PExternal, atom_dict, mol_dict, Sys_dict):

    #Potentials_dict = {}
    print('\n...Setting Bmin to {}'.format(params._Bmin))
    
    for Sys in Sys_dict.values():

        #bonded interactions
        for bond in PBonds:
            atomtypes = sim.atomselect.PolyFilter([atom_dict[bond[0][0]], atom_dict[bond[0][1]]], Bonded=True)
            label = 'bond_{}_{}'.format(bond[0][0], bond[0][1])
            Dist0 = bond[1][0]
            FConst = bond[1][1]
            P = sim.potential.Bond(Sys, Label=label, Filter=atomtypes, Dist0=Dist0, FConst=FConst)
            P.Dist0.Fixed = bond[2][0]
            P.FConst.Fixed = bond[2][1]
            Sys.ForceField.append(P)

        #nonbonded interactions
        for nonbond in PNonbonded:
            atomtypes = sim.atomselect.PolyFilter([atom_dict[nonbond[0][0]], atom_dict[nonbond[0][1]]])
            label = 'ljg_{}_{}'.format(nonbond[0][0], nonbond[0][1])
            B =  nonbond[1][0]
            Kappa = nonbond[1][1]
            P = sim.potential.Gaussian(Sys, Label=label, Filter=atomtypes, Cut=params._rcut, B=B, Kappa=Kappa, Dist0=0.0, Shift=True)
            P.B.Fixed = nonbond[2][0]
            P.Kappa.Fixed = nonbond[2][1]
            P.Dist0.Fixed = True
            P.Param.Min = params._Bmin
            Sys.ForceField.append(P)

        #electrostatic potentials 
        if params._Electrostatics: #turn on ewald summation and smeared coulumb 
            if (params._NeutralSrel and 'ElecSys_' in Sys.Name) or params._NeutralSrel == False:
            
                P = sim.potential.Ewald(Sys, ExcludeBondOrd=0, Cut=params._rcut, Shift=params._Shift, Coef=params._Coef, FixedCoef=True, Label='ewald' )
                Sys.ForceField.append(P)

                for elec in PElectrostatic:
                    atomtypes = sim.atomselect.PolyFilter([atom_dict[elec[0][0]], atom_dict[elec[0][1]]])
                    label = 'smeared_corr_{}_{}'.format(elec[0][0], elec[0][1])
                    Coef=elec[1][0]
                    BornA=elec[1][1]
                    P = sim.potential.SmearedCoulombEwCorr(Sys, Label=label, Filter=atomtypes, Cut=params._rcut, Coef=Coef, BornA=BornA, Shift=params._Shift)
                    P.FixedCoef = elec[2][0]
                    P.FixedBornA = elec[2][1]
                    Sys.ForceField.append(P)
            
        #external potentials
        for external in PExternal:
            if external[0] == 'gaussian':
                atomtypes = sim.atomselect.PolyFilter([atom_dict[external[1][0]], atom_dict[external[1][1]]])
                label = 'ext_gauss_{}_{}'.format(external[1][0], external[1][1])
                B = external[2][0]
                Kappa = external[2][1]
                P = sim.potential.Gaussian(Sys, Label=label, Filter=atomtypes, Cut=params._rcut, B=B, Kappa=Kappa, Dist0=0.0, Shift=True)
                P.Param.Min = -100.
                P.B.Fixed = True
                P.Kappa.Fixed = True
                P.Dist0.Fixed = True
                Sys.ForceField.append(P)

            if external[0] == 'sinusoid':
                atomtypes = sim.atomselect.PolyFilter([atom_dict[external[1][0]]])
                label = 'ext_sin_{}'.format(external[1][0])
                UConst = external[2][0]
                NPeriods = external[2][1]
                PlaneAxis = external[2][2]
                PlaneLoc = external[2][3]
                P = sim.potential.ExternalSinusoid(Sys, Label=label, Filter=atomtypes, Fixed=True, UConst=UConst, NPeriods=NPeriods, PlaneAxis=PlaneAxis, PlaneLoc=PlaneLoc)
                Sys.ForceField.append(P)
                    
        #setup histograms 
        for P in Sys.ForceField:
            P.Arg.SetupHist(NBin = 10000, ReportNBin=100)
            #Potentials_dict[P.Label] = P
        
        if params._ForceFieldFile:
            print('\n...Reading forcefield parameters from {} for {}'.format(params._ForceFieldFile, Sys.Name))
            with open(params._ForceFieldFile, 'r') as of: s = of.read()
            Sys.ForceField.SetParamString(s, CheckMissing = False)

    for Sys_name, Sys in Sys_dict.items():
        print('\nForcefield for {}: '.format(Sys_name))
        for P in Sys.ForceField:
            print(P.Label)

    return Sys_dict

def AddMeasures(Ree2, Area, Sys_dict):

    print('\n========== Adding measures ==========')
    
    if Ree2 == [] and Area == []:
        print('\n... No measures specified')
        return Sys_dict

    for entry in Ree2:
        Sys = Sys_dict[entry[1]]
        mol_name = entry[2]
        site_IDs = entry[3]
        Sys_mol = [mol for mol in Sys.World if mol.Name == mol_name][0]
        site1, site2 = Sys.World.SiteTypes[Sys_mol[site_IDs[0]].SID], Sys.World.SiteTypes[Sys_mol[site_IDs[1]].SID]
        filter = sim.atomselect.PolyFilter(Filters=[site1, site2],Intra=True)
        dist_ree2 = sim.measure.distang.Distance2(Sys, Filter=filter, Name=entry[0])
        Sys.Measures.append(dist_ree2)
        print('{}: added Ree2 measure between sites {} and {}'.format(Sys.Name, site1, site2))

    for entry in Area:
        Sys = Sys_dict[entry[1]]
        Az = sim.measure.Az(Sys,axis=entry[2])
        Sys.Measures.append(Az)
        print('\n{}: added area measure for axis {} '.format(Sys.Name, entry[2]))

    return Sys_dict

def CompileSystems(Sys_dict):

    print('\n========== Loading and locking (compiling) system ==========')
    for Sys in Sys_dict.values():
        Sys.Load()
        
    return Sys_dict

def CreateMapping(params, MolTypes, Systems, Optimizers, Trajectories, Sys_dict):
    
    mapping_dict = {}
    mol_mapping = {}
    for entry in MolTypes:
        mol_mapping[entry[0]] = entry[2]

    system_traj_dict = {}
    trajfile_dict = {}
    for Opt_entry in Optimizers:
        system_traj_dict[Opt_entry[1]] = Opt_entry[2]
    for traj_entry in Trajectories:
        trajfile_dict[traj_entry[0]] = [traj_entry[1],traj_entry[2]]

    for Sys_entry in Systems:
        if params._COM==True:
            trajname = system_traj_dict[Sys_entry[0]]
            traj = md.load(trajfile_dict[trajname][0], top=trajfile_dict[trajname][1])
        Sys = Sys_dict[Sys_entry[0]]
        mapping = sim.atommap.PosMap()
        index=0 
        nmol = 0
        j = 0
        k = 0
        molmap = mol_mapping[Sys_entry[1][j]]
        for i, atom in enumerate(Sys.Atom):
            #print(list(range(index,index+molmap[k])),atom)
            aa_indices = list(range(index,index+molmap[k]))
            if params._COM==True: masses = np.array([traj.top.atom(ia).element.mass for ia in aa_indices])
            else: masses=None
            #print(masses)
            mapping.Add(Atoms1=aa_indices, Atom2=atom, Mass1=masses)
            index += molmap[k]
        
            if k == len(molmap) - 1: 
                nmol += 1
                k = 0
            else: k += 1
            
            if nmol == Sys_entry[2][j]:
                if j + 1 < len(Sys_entry[2]):
                    nmol = 0
                    j += 1
                    molmap = mol_mapping[Sys_entry[1][j]]

        mapping_dict[Sys_entry[0]] = mapping
        
    return mapping_dict

def LoadTrajectories(Trajectories):
    
    traj_dict = {}
    for traj_entry in Trajectories:
        if traj_entry[3] == 'dcd':
            traj = sim.traj.dcd.DCD(TrjFile=traj_entry[1], TopFile=traj_entry[2], ConversionFactor=1)
        elif traj_entry[3] == 'pdb':
            traj = sim.traj.pdb.Pdb(PdbFile=traj_entry[1])
        elif traj_entry[3] == 'lammpstrj':
            traj = sim.traj.lammps.Lammps(TrjFile=traj_entry[1])
        elif traj_entry[3] == 'xyz':
            traj = sim.traj.xyz.XYZ(TrjFile=traj_entry[1])
        traj_dict[traj_entry[0]] = traj

    return traj_dict

def CreateOptimizer(params, Optimizers, Sys_dict, traj_dict, mapping_dict):

    print('\n========== Making optimizer ==========')

    if params._MDEngine == 'openmm':
        OptClass = sim.srel.optimizetrajomm.OptimizeTrajOpenMMClass
    elif params._MDEngine == 'sim':
        OptClass = sim.srel.optimizetraj.OptimizeTrajClass

    Opt_dict = {}
    for Opt_entry in Optimizers:
        Sys_name = Opt_entry[1]
        traj_name = Opt_entry[2]
        Sys = Sys_dict[Sys_name]
        mapping = mapping_dict[Sys_name]
        traj = traj_dict[traj_name]
        Sys.Pos = sim.traj.Mapped(traj, mapping, Sys=Sys)[0]
        #print(Sys.Pos)
       
        if 0.0 in Sys.BoxL: UseTrajBoxL=True
        else: UseTrajBoxL=False
        if params._Electrostatics and params._NeutralSrel: ElecSys = Sys_dict['ElecSys_{}'.format(Sys_name)]
        else: ElecSys = None
        Optimizer = OptClass(ModSys=Sys, Map=mapping, Traj=traj, 
            FilePrefix=Opt_entry[3], LoadArgData=True, Verbose=True, UseTrajBoxL=UseTrajBoxL, ElecSys=ElecSys)
        Optimizer.StepsEquil = params._StepsEquil
        Optimizer.StepsProd = params._StepsProd
        Optimizer.StepsStride = params._StepsStride
        Optimizer.HessianAdjustMode = 2
        #Optimizer.UseHessian = False
        Opt_dict[Opt_entry[0]] = Optimizer

        print('\nOptimizer Specifications:')
        print('File Prefix: {}'.format(Optimizer.FilePrefix))
        print('Model System: {}'.format(Optimizer.ModSys.Name))
        if Optimizer.ElecSys != None: print('ElecSys: {}'.format(Optimizer.ElecSys.Name))
        print('NMol: {}'.format(Optimizer.ModSys.NMol))
        print('NAtom: {}'.format(Optimizer.ModSys.NAtom))
        print('NDOF: {}'.format(Optimizer.ModSys.NDOF))
        print('Verbose: {}'.format(Optimizer.Verbose))
        
    return Opt_dict

def AddPenalties(Penalties, Opt_dict):

    print('\n========== Adding penalties ==========')

    if Penalties == []:
        print('\n... No pentalties specified')
        return Opt_dict
    
    for entry in Penalties:
        Opt= Opt_dict[entry[0]]
        measure = [measure for measure in Opt.ModSys.Measures if measure.Name == entry[1]][0]
        target = entry[2]
        Opt.AddPenalty(measure, target, MeasureScale = 1., Coef=1.e-80)
        print('\n{}: added {} penalty with target value: {}'.format(Opt.FilePrefix, measure.Name, target))

    return Opt_dict

def Run(Runs, Opt_dict):

    for entry in Runs:
        Opts = [Opt_dict[name] for name in entry[0]]
        mode = entry[1]
        for Opt in Opts: Opt.UpdateMode = mode
        if len(Opts) == 1:
            Opt = Opts[0]
            Sys = Opt.ModSys
            if mode != 3: 
                print('\n========== RUNNING SREL ==========')
                if mode == 1:   
                    Opt.Run()
                elif mode == 0 or mode == 2:
                    StageCoefs = entry[2]
                    Opt.RunStages(StageCoefs = StageCoefs, UseLagMult = 10.0)

                Opt.OutputPotentials(FilePrefix = 'final')
                if Opt.ElecSys != None: 
                    ParamString = Opt.ElecSys.ForceField.ParamString()
                    paramfile = open(Opt.FilePrefix+'_ElecSys_ff.dat','w')
                    paramfile.write(ParamString)
                    paramfile.close()

                SaveHessian(Opt)

            elif mode == 3:
                print('\n========== RUNNING CGMD ==========')
                Opt.MakeModTrajFunc(Opt, Sys, Opt.FilePrefix, Opt.StepsEquil, Opt.StepsProd, Opt.StepsStride, Verbose=True)

        elif len(Opts) > 1:
            weights = entry[3]
            file_prefix = '_'.join([Opt.FilePrefix for Opt in Opts])
            OptMulti = sim.srel.optimizemultitraj.OptimizeMultiTrajClass(Opts ,Weights=weights,FilePrefix=file_prefix)
            if mode != 3: 
                print('\n========== RUNNING SREL ==========')
                OptMulti.UpdateMode = mode 
                if mode == 1:   
                    OptMulti.Run()
                elif mode == 0 or mode == 2:
                    StageCoefs = entry[2]
                    OptMulti.RunStages(StageCoefs = StageCoefs, UseLagMult = 10.0)

                OptMulti.OutputPotentials(FilePrefix = 'final')
                if OptMulti.OptimizeTrajList[0].ElecSys != None: 
                    ParamString = OptMulti.OptimizeTrajList[0].ElecSys.ForceField.ParamString()
                    paramfile = open(Opt.FilePrefix+'_ElecSys_ff.dat','w')
                    paramfile.write(ParamString)
                    paramfile.close()

                for Opt in OptMulti.OptimizeTrajList:
                    SaveHessian(Opt)

            elif mode == 3:
                for Opt in Opts:
                    Sys = Opt.ModSys
                    print('\n========== RUNNING CGMD: {} =========='.format(Sys.Name))
                    Opt.MakeModTrajFunc(Opt, Sys, Opt.FilePrefix, Opt.StepsEquil, Opt.StepsProd, Opt.StepsStride, Verbose=True)

    return
    
#========== Execution ==========#

def execute():
    
    params = Parameters(COM, ForceFieldFile, asmear, rcut, Bmin, Electrostatics, NeutralSrel, Coef, BornA, Shift, Temp, Pressure, Axis, Tension, dt, LangevinGamma, MDEngine, StepsEquil, StepsProd, StepsStride)
    atom_dict, mol_dict, Sys_dict = CreateSystem(params, Systems, AtomTypes, MolTypes, Bonds)
    Sys_dict = SetupPotentials(params, PBonds, PNonbonded, PElectrostatic, PExternal, atom_dict, mol_dict, Sys_dict)
    Sys_dict = AddMeasures(Ree2, Area, Sys_dict)
    Sys_dict = CompileSystems(Sys_dict)
    mapping_dict = CreateMapping(params, MolTypes, Systems, Optimizers, Trajectories, Sys_dict)
    traj_dict = LoadTrajectories(Trajectories)
    Opt_dict = CreateOptimizer(params, Optimizers, Sys_dict, traj_dict, mapping_dict)
    Opt_dict = AddPenalties(Penalties, Opt_dict)
    Run(Runs, Opt_dict)    


if __name__ == "__main__":

    parser = ap.ArgumentParser(description='Simulation Parameters')
    parser.add_argument("paramfile", default='sim_params.py', type=str, help="sim_params.py file")
    cmdln_args = parser.parse_args()

    #initialize defaults for simulation parameters
    COM=True 
    ForceFieldFile=None
    asmear=0.5
    rcut=1.0
    Bmin=0.0
    Electrostatics=False
    NeutralSrel=False 
    Coef=None 
    BornA=None 
    Shift=True
    Temp=1.0
    Pressure=0.0 
    Tension=None,
    Axis=None 
    Tension=None 
    dt=0.1 
    LangevinGamma=1.0 
    MDEngine='openmm'
    StepsEquil=0 
    StepsProd=0 
    StepsStride=0

    # initialize default empty arrays
    AtomTypes, MolTypes, Bonds = [], [], []
    PBonds, PNonbonded, PElectrostatic, PExternal = [], [], [], []
    Systems, Trajectories, Optimizers, Runs = [], [], [], []
    Ree2, Area, Penalties = [], [], []

    # execute the parameter file code
    with open(cmdln_args.paramfile, 'r') as of:
        filecode = of.read()
    exec(filecode, None, globals())

    # main function call
    execute()