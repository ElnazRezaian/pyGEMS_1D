import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" # don't print all the TensorFlow warnings
import perform.constants as const
from perform.systemSolver import systemSolver
from perform.solution.solutionDomain import solutionDomain
from perform.visualization.visualizationGroup import visualizationGroup
from perform.rom.romDomain import romDomain
from perform.miscFuncs import mkdirInWorkdir

import numpy as np
import argparse
import traceback
import warnings
warnings.filterwarnings("error")
from time import time

# TODO: make code general for more than two species, array broadcasts are different for 2 vs 3+ species
#		idea: custom iterators for species-related slicing, or just squeeze any massfrac references
# TODO: check all calls to calcDensityDerivatives and calcStagEnthalpyDerivatives, pass full massFrac array

def main():

	##### START SETUP #####

	# read working directory input
	parser = argparse.ArgumentParser(description = "Read working directory")
	parser.add_argument('workingDir', type = str, default = "./", help="runtime working directory")
	const.workingDir = os.path.expanduser(parser.parse_args().workingDir)
	assert (os.path.isdir(const.workingDir)), "Given working directory does not exist"

	# make output directories
	const.unsteadyOutputDir = mkdirInWorkdir(const.unsteadyOutputDirName)
	const.probeOutputDir = mkdirInWorkdir(const.probeOutputDirName)
	const.imageOutputDir = mkdirInWorkdir(const.imageOutputDirName)
	const.restartOutputDir = mkdirInWorkdir(const.restartOutputDirName)

	solver = systemSolver()				# mesh, gas model, and time integrator
										# TODO: multi-domain solvers

	# physical and ROM solutions
	solDomain = solutionDomain(solver)	
	if solver.calcROM:
		rom = romDomain(solDomain, solver)
	else:
		rom = None

	visGroup = visualizationGroup(solDomain, solver) # plots

	##### END SETUP #####

	##### START UNSTEADY SOLUTION #####

	try:
		# loop over time iterations
		t1 = time()
		for solver.iter in range(1, solver.numSteps+1):
			
			# advance one physical time step
			if (solver.calcROM):
				rom.advanceIter(solDomain, solver)
			else:
				solDomain.advanceIter(solver)
			solver.timeIter += 1
			solver.solTime  += solver.dt

			# write unsteady solution outputs
			solDomain.writeIterOutputs(solver)

			# check "steady" solve
			if (solver.runSteady):
				breakFlag = solDomain.writeSteadyOutputs(solver)
				if breakFlag: break

			# visualization
			visGroup.drawPlots(solDomain, solver)

		runtime = time() - t1
		print("Solve finished in %.8f seconds, writing to disk" % runtime)

	except RuntimeWarning:
		solver.solveFailed = True
		print(traceback.format_exc())
		print("Solve failed, dumping solution so far to disk")

	##### END UNSTEADY SOLUTION #####

	##### START POST-PROCESSING #####

	solDomain.writeFinalOutputs(solver)

	##### END POST-PROCESSING #####

if __name__ == "__main__":
	try:
		main()
	except:
		print(traceback.format_exc())
		print("Execution failed")