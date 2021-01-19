from pygems1d.constants import realType, RUniv

import numpy as np

# TODO: some of the CPG functions can be generalized and placed here (e.g. calc sound speed in terms of enthalpy and density derivs) 

class gasModel:
	"""
	Base class storing constant chemical properties of modeled species
	Also includes universal gas methods (like calculating mixture molecular weight)
	"""

	def __init__(self, gasDict):

		# gas composition
		self.numSpeciesFull 	= int(gasDict["numSpecies"])				# total number of species in case
		self.molWeights 		= gasDict["molWeights"].astype(realType)	# molecular weights, g/mol
		self.enthRef 			= gasDict["enthRef"].astype(realType) 		# reference enthalpy, J/kg
		self.tempRef 			= gasDict["tempRef"]						# reference temperature, K
		self.Cp 				= gasDict["Cp"].astype(realType)			# heat capacity at constant pressure, J/(kg-K)
		self.Pr 				= gasDict["Pr"].astype(realType)			# Prandtl number
		self.Sc 				= gasDict["Sc"].astype(realType)			# Schmidt number
		self.muRef				= gasDict["muRef"].astype(realType)			# reference dynamic viscosity for Sutherland model
		
		# Arrhenius factors
		# TODO: modify these to allow for multiple global reactions
		self.nu 				= gasDict["nu"].astype(realType)		# ?????
		self.nuArr 				= gasDict["nuArr"].astype(realType)		# ?????
		self.actEnergy			= float(gasDict["actEnergy"])			# global reaction Arrhenius activation energy, divided by RUniv, ?????
		self.preExpFact 		= float(gasDict["preExpFact"]) 			# global reaction Arrhenius pre-exponential factor		

		# misc calculations
		self.RGas 				= RUniv / self.molWeights 			# specific gas constant of each species, J/(K*kg)
		self.numSpecies 		= self.numSpeciesFull - 1			# last species is not directly solved for
		self.numEqs 			= self.numSpecies + 3				# pressure, velocity, temperature, and species transport
		self.molWeightNu 		= self.molWeights * self.nu 
		self.mwInvDiffs 		= (1.0 / self.molWeights[:-1]) - (1.0 / self.molWeights[-1]) 
		self.CpDiffs 			= self.Cp[:-1] - self.Cp[-1]
		self.enthRefDiffs 		= self.enthRef[:-1] - self.enthRef[-1]

		# mass matrices for calculating viscosity and thermal conductivity mixing laws
		self.mixMassMatrix 		= np.zeros((self.numSpeciesFull, self.numSpeciesFull), dtype=realType)
		self.mixInvMassMatrix 	= np.zeros((self.numSpeciesFull, self.numSpeciesFull), dtype=realType)
		self.precompMixMassMatrices()

	def precompMixMassMatrices(self):
		for specNum in range(self.numSpeciesFull):
			self.mixMassMatrix[specNum, :] 		= np.power((self.molWeights / self.molWeights[specNum]), 0.25)
			self.mixInvMassMatrix[specNum, :] 	= 1.0 / np.sqrt( 1.0 + self.molWeights[specNum] / self.molWeights)

	def getMassFracArray(self, solPrim=None, massFracs=None):
		"""
		Helper function to handle array slicing to avoid weird NumPy array broadcasting issues
		"""
		# get all but last mass fraction field
		if (solPrim is None):
			assert (massFracs is not None), "Must provide mass fractions if not providing primitive solution"
			if (massFracs.ndim == 1):
				massFracs = np.reshape(massFracs, (1,-1))
			if (massFracs.shape[1] == self.numSpeciesFull):
				massFracs = massFracs[:-1,:]
		else:
			massFracs = solPrim[3:,:]
		print('get mass shape',massFracs.shape)
		# slice array appropriately
		if (self.numSpecies > 1):
			massFracs = massFracs[:-1, :]
		else:
			massFracs = massFracs[0,:]

		return massFracs

	def calcAllMassFracs(self, massFracsNS):
		"""
		Helper function to compute all numSpecies_full mass fraction fields from numSpecies fields
		Thresholds all mass fraction fields between zero and unity 
		"""

		numSpecies, numCells = massFracsNS.shape
		massFracs = np.zeros((numSpecies+1,numCells), dtype=realType)

		massFracs[:-1,:] 	= np.maximum(0.0, np.minimum(1.0, massFracsNS))
		massFracs[-1,:] 	= 1.0 - np.sum(massFracs[:-1,:], axis=0)
		massFracs[-1,:] 	= np.maximum(0.0, np.minimum(1.0, massFracs[-1,:]))

		return massFracs

	def calcMixMolWeight(self, massFracs):
		"""
		Compute mixture molecular weight
		"""

		if (massFracs.shape[0] == self.numSpecies):
			massFracs = calcAllMassFracs(massFracs)

		mixMolWeight = 1.0 / np.sum(massFracs / self.molWeights, axis=0)

		return mixMolWeight
	