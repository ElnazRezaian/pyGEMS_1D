from perform.constants import realType

import numpy as np

def calcCellGradients(solDomain, solver):
	"""
	Compute cell-centered gradients for higher-order face reconstructions
	Also calculate gradient limiters if requested
	"""

	# compute gradients via finite difference stencil
	solPrimGrad = np.zeros((solDomain.gasModel.numEqs, solDomain.numGradCells), dtype=realType)
	if (solver.spaceOrder == 2):
		solPrimGrad = (0.5 / solver.mesh.dx) * (solDomain.solPrimFull[:, solDomain.gradIdxs + 1] - solDomain.solPrimFull[:, solDomain.gradIdxs - 1])
	else:
		raise ValueError("Order "+str(solver.spaceOrder)+" gradient calculations not implemented")

	# compute gradient limiter and limit gradient, if requested
	if (solver.gradLimiter != ""):

		# Barth-Jespersen
		if (solver.gradLimiter == "barth"):
			phi = limiterBarthJespersen(solDomain, solPrimGrad, solver.mesh)

		# Venkatakrishnan, no correction
		elif (solver.gradLimiter == "venkat"):
			phi = limiterVenkatakrishnan(solDomain, solPrimGrad, solver.mesh)

		else:
			raise ValueError("Invalid input for gradLimiter: "+str(solver.gradLimiter))

		solPrimGrad = solPrimGrad * phi	# limit gradient

	return solPrimGrad
	

def findNeighborMinMax(sol):
	"""
	Find minimum and maximum of cell state and neighbor cell state
	"""

	# max and min of cell and neighbors
	solMax = sol.copy()
	solMin = sol.copy()

	# first compare against right neighbor
	solMax[:,:-1]  	= np.maximum(sol[:,:-1], sol[:,1:])
	solMin[:,:-1]  	= np.minimum(sol[:,:-1], sol[:,1:])

	# then compare agains left neighbor
	solMax[:,1:] 	= np.maximum(solMax[:,1:], sol[:,:-1])
	solMin[:,1:] 	= np.minimum(solMin[:,1:], sol[:,:-1])

	return solMin, solMax


def limiterBarthJespersen(solDomain, grad, mesh):
	"""
	Barth-Jespersen limiter
	Ensures that no new minima or maxima are introduced in reconstruction
	"""

	solPrim = solDomain.solPrimFull[:, solDomain.gradIdxs]

	# get min/max of cell and neighbors
	solPrimMin, solPrimMax = findNeighborMinMax(solDomain.solPrimFull[:, solDomain.gradNeighIdxs])

	# extract gradient cells
	solPrimMin = solPrimMin[:, solDomain.gradNeighExtract]
	solPrimMax = solPrimMax[:, solDomain.gradNeighExtract]

	# unconstrained reconstruction at neighboring cell centers
	delSolPrim 		= grad * mesh.dx
	solPrimL 		= solPrim - delSolPrim
	solPrimR 		= solPrim + delSolPrim
	
	# limiter defaults to 1
	phiL = np.ones(solPrim.shape, dtype=realType)
	phiR = np.ones(solPrim.shape, dtype=realType)
	
	# find indices where difference in reconstruction is either positive or negative
	cond1L = ((solPrimL - solPrim) > 0)
	cond1R = ((solPrimR - solPrim) > 0)
	cond2L = ((solPrimL - solPrim) < 0)
	cond2R = ((solPrimR - solPrim) < 0)

	# threshold limiter for left and right reconstruction
	phiL[cond1L] = np.minimum(1.0, (solPrimMax[cond1L] - solPrim[cond1L]) / (solPrimL[cond1L] - solPrim[cond1L]))
	phiR[cond1R] = np.minimum(1.0, (solPrimMax[cond1R] - solPrim[cond1R]) / (solPrimR[cond1R] - solPrim[cond1R]))
	phiL[cond2L] = np.minimum(1.0, (solPrimMin[cond2L] - solPrim[cond2L]) / (solPrimL[cond2L] - solPrim[cond2L]))
	phiR[cond2R] = np.minimum(1.0, (solPrimMin[cond2R] - solPrim[cond2R]) / (solPrimR[cond2R] - solPrim[cond2R]))

	# take minimum limiter from left and right
	phi = np.minimum(phiL, phiR)
	
	return phi


def limiterVenkatakrishnan(solDomain, grad, mesh):
	"""
	Venkatakrishnan limiter
	Differentiable, but limits in uniform regions
	"""

	solPrim = solDomain.solPrimFull[:, solDomain.gradIdxs]

	# get min/max of cell and neighbors
	solPrimMin, solPrimMax = findNeighborMinMax(solDomain.solPrimFull[:, solDomain.gradNeighIdxs])

	# extract gradient cells
	solPrimMin = solPrimMin[:, solDomain.gradNeighExtract]
	solPrimMax = solPrimMax[:, solDomain.gradNeighExtract]

	# unconstrained reconstruction at neighboring cell centers
	delSolPrim 		= grad * mesh.dx
	solPrimL 		= solPrim - delSolPrim
	solPrimR 		= solPrim + delSolPrim
	
	# limiter defaults to 1
	phiL = np.ones(solPrim.shape, dtype=realType)
	phiR = np.ones(solPrim.shape, dtype=realType)
	
	# find indices where difference in reconstruction is either positive or negative
	cond1L = ((solPrimL - solPrim) > 0)
	cond1R = ((solPrimR - solPrim) > 0)
	cond2L = ((solPrimL - solPrim) < 0)
	cond2R = ((solPrimR - solPrim) < 0)
	
	# (y^2 + 2y) / (y^2 + y + 2)
	def venkatakrishnanFunction(maxVals, cellVals, faceVals):
		frac = (maxVals - cellVals) / (faceVals - cellVals)
		fracSq = np.square(frac)
		venkVals = (fracSq + 2.0 * frac) / (fracSq + frac + 2.0)
		return venkVals

	# apply smooth Venkatakrishnan function
	phiL[cond1L] = venkatakrishnanFunction(solPrimMax[cond1L], solPrim[cond1L], solPrimL[cond1L]) 
	phiR[cond1R] = venkatakrishnanFunction(solPrimMax[cond1R], solPrim[cond1R], solPrimR[cond1R]) 
	phiL[cond2L] = venkatakrishnanFunction(solPrimMin[cond2L], solPrim[cond2L], solPrimL[cond2L])
	phiR[cond2R] = venkatakrishnanFunction(solPrimMin[cond2R], solPrim[cond2R], solPrimR[cond2R])

	# take minimum limiter from left and right
	phi = np.minimum(phiL, phiR)

	return phi