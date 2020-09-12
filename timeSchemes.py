from solution import solutionPhys, boundaries
from romClasses import solutionROM
from classDefs import parameters, geometry, gasProps
from spaceSchemes import calcRHS
from boundaryFuncs import calcBoundaries
from Jacobians import calc_dresdsolPrim, calc_dresdsolPrim_imag, calc_dsolConsdsolPrim_imag, vec_assemble
import constants
from scipy.sparse.linalg import spsolve
import numpy as np
import pdb


def init_sol_mat(sol, bounds, params, gas):
	
	sol_mat = []
	
	calcBoundaries(sol, bounds, params, gas)
	
	for i in range(params.timeOrder+1):
		sol_mat.append(sol.solCons.copy())
	
	return sol_mat


def update_sol_mat(sol_mat):
	
	sol_mat[1:] = sol_mat[:-1]
			
	return sol_mat

# TODO: cold start is not valid for timeOrder > 2
def calc_implicitres(sol: solutionPhys, sol_mat, bounds: boundaries, params: parameters, geom: geometry, gas: gasProps, colstrt):
	
	t_or = params.timeOrder
	
	if (colstrt): # cold start
		params.timeOrder = 1 
	
	calcRHS(sol, bounds, params, geom, gas) # non-linear RHS of current solution
	
	if (params.timeOrder == 1):
		res = (sol.solCons - sol_mat[1])/(params.dt)
	elif (params.timeOrder == 2):
		res = (1.5*sol.solCons - 2.*sol_mat[1] + 0.5*sol_mat[2])/(params.dt)
	elif (params.timeOrder == 3):
		res = (11./6.*sol.solCons - 3.*sol_mat[1] + 1.5*sol_mat[2] -1./3.*sol_mat[3])/(params.dt)
	elif (params.timeOrder == 4):
		res = (25./12.*sol.solCons - 4.*sol_mat[1] + 3.*sol_mat[2] -4./3.*sol_mat[3] + 0.25*sol_mat[4])/(params.dt)
	else:
		raise ValueError("Implicit Schemes higher than BDF4 not-implemented")
	
	res = -res + sol.RHS
	params.timeOrder = t_or
	
	return res

# explicit time integrator, one subiteration
def advanceexplicit(sol: solutionPhys, rom: solutionROM, 
					bounds: boundaries, params: parameters, geom: geometry, gas: gasProps, 
					subiter, solOuter):
	
	
	#compute RHS function
	calcRHS(sol, bounds, params, geom, gas)
	
	# if solPrim, calculate d(solPrim)/dt
	# if params.solforPrim:
	# 	sol.RHS = calc_dsolPrim(sol, gas) 
		
	pdb.set_trace()
	# compute change in solution/code, advance solution/code
	# if (params.calcROM)
	rom.mapRHSToModels(sol)
	rom.calcRHSProjection()
	dSol_rom = rom.advanceSubiter(sol, params, subiter)
	# else:
	dSol = params.dt * params.subIterCoeffs[subiter] * sol.RHS
	
	pdb.set_trace()

	# if ROM, reconstruct solution
	
	# update state
	# if params.solforPrim:
	# 	sol.solPrim = solOuter + dSol
	# 	sol.updateState(gas, fromCons = False)  
	# else:
	sol.solCons = solOuter + dSol
	sol.updateState(gas)
	
	return sol
   
# implicit pseudo-time integrator, one subiteration
def advancedual(sol, sol_mat, bounds, params, geom, gas, colstrt=False):
	

	# residual
	res = calc_implicitres(sol, sol_mat, bounds, params, geom, gas, colstrt)
	
	# dt_inv
	dt_inv = constants.bdfCoeffs[params.timeOrder-1]/params.dt
	dtau_inv = 1./params.dtau


	# Jacobian
	resJacob = calc_dresdsolPrim(sol, gas, geom, params, bounds, dt_inv, dtau_inv)
	
	# Comparing with numerical jacobians
	# diff = calc_dresdsolPrim_imag(sol, gas, geom, params, bounds, dt_inv, dtau_inv)
	# print(diff)

	# Solving linear system 
	resJacob_sparse = vec_assemble(resJacob)
	dSol = spsolve(resJacob_sparse, (res.flatten('F')))

	# updating state
	sol.solPrim += dSol.reshape((geom.numCells, gas.numEqs), order='F')
	sol.updateState(gas, fromCons = False)
	
	res = dSol.reshape((geom.numCells, gas.numEqs), order = 'F')
	
	return sol_mat, res

	
	