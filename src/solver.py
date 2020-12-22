
import numpy as np
import matplotlib.pyplot as plt
import copy
from classDefs import parameters, geometry, gasProps
from solution import solutionPhys, boundaries, genInitialCondition
from spaceSchemes import calcRHS
from stateFuncs import calcStateFromPrim
from romClasses import solutionROM
from timeSchemes import advanceExplicit, advanceDual
from inputFuncs import readRestartFile
import outputFuncs
import constants
import time
import os
import sys
import pdb

# driver function for advancing the solution
#@profile
def solver(params: parameters, geom: geometry, gas: gasProps):

	# TODO: could move most of this to driver? Or just move to one big init function
	# TODO: make an option to interpolate a solution onto the given mesh, if different
	# intialize from restart file
	if params.initFromRestart:
		params.solTime, solPrim0 = readRestartFile(params.restOutDir)
		solCons0, _, _, _ = calcStateFromPrim(solPrim0, gas)

	# otherwise init from scratch IC or custom IC file 
	else:
		if (params.initFile == None):
			solPrim0, solCons0 = genInitialCondition(params, gas, geom)
		else:
			# TODO: change this to .npz format with physical time included
			solPrim0 = np.load(params.initFile)
			solCons0, _, _, _ = calcStateFromPrim(solPrim0, gas)
	sol = solutionPhys(geom.numCells, solPrim0, solCons0, gas, params)
	
	# add bulk velocity if required
	# TODO: should definitely be moved somewhere else
	if (params.velAdd != 0.0):
		sol.solPrim[:,1] += params.velAdd
	
	sol.updateState(gas, fromCons = False)

	# initialize ROM
	if params.calcROM: 
		rom = solutionROM(params.romInputs, sol, params)
		rom.initializeROMState(sol)
	else:
		rom = None

	# initialize boundary state
	bounds = boundaries(sol, params, gas)

	# prep probe
	# TODO: expand to multiple probe locations
	probeIdx = 0
	if (params.probeLoc > geom.xR):
		params.probeSec = "outlet"
	elif (params.probeLoc < geom.xL):
		params.probeSec = "inlet"
	else:
		params.probeSec = "interior"
		probeIdx = np.absolute(geom.xCell - params.probeLoc).argmin()
	probeVals = np.zeros((params.numSteps, params.numVis), dtype = constants.realType)

	# prep visualization
	if (params.visType != "None"): 
		fig, ax, axLabels = outputFuncs.setupPlotAxes(params)
		visName = ""
		for visVar in params.visVar:
			visName += "_"+visVar
		visName += "_"+params.simType

	tVals = np.linspace(params.dt, params.dt*params.numSteps, params.numSteps, dtype = constants.realType)
	if ((params.visType == "field") and params.visSave):
		fieldImgDir = os.path.join(params.imgOutDir, "field"+visName)
		if not os.path.isdir(fieldImgDir): os.mkdir(fieldImgDir)
	else:
		fieldImgDir = None

	# initializing time history for implicit schemes
	if (params.timeType == "implicit"): sol.initSolHist(params) 

	# loop over time iterations
	for tStep in range(params.numSteps):
		
		if (not params.runSteady): print("Iteration "+str(tStep+1))

		# time integration scheme, advance one physical time step
		advanceSolution(sol, rom, bounds, params, geom, gas)

		if (params.timeType == "implicit"):
			# updating time history
			sol.updateSolHist() 

			# print norm of change in solution
			if (params.runSteady):
				sol.resOutput(params, tStep)

		params.solTime += params.dt

		# write restart files
		if params.saveRestarts: 
			if ( ((tStep+1) % params.restartInterval) == 0):
				outputFuncs.writeRestartFile(sol, params, tStep)	 

		# write output
		if ( (((tStep+1) % params.outInterval) == 0) ):
			outputFuncs.storeFieldDataUnsteady(sol, params, tStep)
			if (params.runSteady): outputFuncs.writeDataSteady(sol, params)
		
		outputFuncs.updateProbe(sol, params, bounds, probeVals, probeIdx, tStep)
		if (params.runSteady): 
			outputFuncs.updateResOut(sol, params, tStep)
			if (sol.resOutL2 < params.steadyThresh): 
				print("Steady residual criterion met, terminating run...")
				break 	# quit if steady residual threshold met

		# draw visualization plots
		if ( ((tStep+1) % params.visInterval) == 0):
			if (params.visType == "field"): 
				outputFuncs.plotField(fig, ax, axLabels, sol, params, geom)
				if params.visSave: outputFuncs.writeFieldImg(fig, params, tStep, fieldImgDir)
			elif (params.visType == "probe"): 
				outputFuncs.plotProbe(fig, ax, axLabels, sol, params, probeVals, tStep, tVals)
			
	print("Solve finished, writing to disk!")

	# write data to disk
	outputFuncs.writeDataUnsteady(sol, params, probeVals, tVals)
	if (params.runSteady): outputFuncs.writeDataSteady(sol, params)

	# draw images, save to disk
	if ((params.visType == "probe") and params.visSave): 
		figFile = os.path.join(params.imgOutDir,"probe"+visName+".png")
		fig.savefig(figFile)


# numerically integrate ODE forward one physical time step
def advanceSolution(sol: solutionPhys, rom: solutionROM, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps):
    
	# explicit time integrator is low-mem, only uses outer loop
	# TODO: just make explicit integrator interoperable with solHist
	if (params.timeType == "explicit"): 
		if (params.calcROM):
			solOuter = rom.getCode()
		else:
			solOuter = sol.solCons.copy()

	# loop over max subiterations
	for subiter in range(params.numSubIters):
              
		if (params.timeType == "explicit"):  
   			sol = advanceExplicit(sol, rom, bounds, params, geom, gas, subiter, solOuter)
                       
		else:  

			# TODO: definitely a better way to work the cold start, gotta make operable with timeOrder > 2
			if (params.solTime <= params.timeOrder*params.dt): 
				advanceDual(sol, bounds, params, geom, gas, colstrt=True) # cold-start          
			else:
				advanceDual(sol, bounds, params, geom, gas)
       			 
			# checking sub-iterations convergence
			resNorm = np.linalg.norm(sol.res, ord=2)
			if (not params.runSteady): print(resNorm)
			if (resNorm < params.resTol): break
		