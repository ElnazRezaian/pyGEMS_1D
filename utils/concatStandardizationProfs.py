# combines separate standardization profiles (e.g. from "scalar" ROM treatment), concatenates them together

import numpy as np
import os
import pdb

##### BEGIN USER INPUT #####

dataDir = "/home/chris/Research/GEMS_runs/prf_nonlinManifold/pyGEMS/standingFlame/dataProc/tfModels/CAE_k3_scalar" # directory where all standardization profiles are held

standProfiles = ["normSub_Density.npy","normSub_Momentum.npy","normSub_Enthalpy.npy","normSub_Rho-Reactant_mf.npy"] # subpaths to separate standardization profiles

# zero-indexed hash table between ordering of standardization profiles and ordering of governing variables
# conservative ordering: density, momentum, enthalpy, density-weighted mass fractions
# primitive ordering: pressure, velocity, temperature, mass fractions
varOrder = [0,1,2,3]

profOutName = "normSub_full" 

# EXAMPLE
# Two standardization profiles: standProfiles = ["norm_pressure_YCH4", "norm_velocity_temperature"]
# standProfiles will be concatenated in given order, so loaded order will be pressure, YCH4, velocity, temperature
# need to switch order to correct primitive ordering, so varOrder = [0,2,3,1]

##### END USER INPUT #####

# aggregate standardization profiles
for profIdx, prof in enumerate(standProfiles):

	fileIn = os.path.join(dataDir, prof)
	dataIn = np.load(fileIn)

	if (dataIn.ndim == 1): dataIn = dataIn[:,None] 

	if (profIdx == 0):
		dataOut = dataIn.copy()
	else:
		dataOut = np.concatenate((dataOut, dataIn), axis=1)

# make sure correct number of entries in varOrder
try:
	assert(len(varOrder) == dataOut.shape[-1])
except AssertionError as ex:
	print("Mismatched size of varOrder ("+str(len(varOrder))+") and number of profile variables ("+str(dataOut.shape[-1])+")")
	raise ex

# write to file
dataOut = dataOut[:,varOrder]
fileOut = os.path.join(dataDir, profOutName+".npy")
np.save(fileOut, dataOut)

print("Finished!")