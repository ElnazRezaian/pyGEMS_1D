import os
import numpy as np
from inputFuncs import readInputFile
from classDefs import parameters, gasProps, catchInput
from solution import solutionPhys
from constants import realType
from Jacobians import calcAnalyticalTFJacobian
import tensorflow as tf 
from tensorflow.keras.models import load_model
from scipy.linalg import pinv
import matplotlib.pyplot as plt
import pdb

# TODO: a little too liberal with similar variable names between solutionROM and model
# TODO: quite a bit changes if using primitive variables or conservative variables
# 		This includes VTROM, and which variables are projected
# TODO: this module is obscenely messy. Move some functions to separate file

# overarching class containing all info/objects/methods required to compute ROMs
class solutionROM:

	def __init__(self, sol: solutionPhys, params: parameters):

		# read input parameters
		romDict = readInputFile(params.romInputs)

		# ROM methods
		self.romMethod 		= catchInput(romDict, "romMethod", "linear") 		# accepts "linear" or "nonlinear"
		if (self.romMethod == "nonlinear"):
			# make sure TF doesn't gobble up device memory
			gpus = tf.config.experimental.list_physical_devices('GPU')
			if gpus:
				try:
					# Currently, memory growth needs to be the same across GPUs
					for gpu in gpus:
						tf.config.experimental.set_memory_growth(gpu, True)
						logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				except RuntimeError as e:
					# Memory growth must be set before GPUs have been initialized
					print(e)
			self.encoderApprox = catchInput(romDict, "encoderApprox", False)	# whether to make encoder projection approximation
		else:
			self.encoderApprox = False
		self.romProj 		= catchInput(romDict, "romProj", "galerkin") 		# accepts "galerkin" or "lspg"
		params.simType 		= self.romMethod + "_" + self.romProj

		# model parameters
		self.modelDir 		= romDict["modelDir"]		# location of all models
		self.numModels 		= romDict["numModels"]		# number of models (trial bases/decoders) being used
		self.modelNames 	= romDict["modelNames"] 	# single string for linear basis, list of strings for NLM
														# FORMAT GUIDE: 
														#	For linear: full name of NPY binary, excluding file extension (assumed *.npy)
														# 	For NLM: all text after "decoder_"/"encoder_", excluding file extension (assumed *.h5)
		self.latentDims 	= romDict["latentDims"]		# list of latent code dimensions for each model
		self.modelVarIdxs 	= romDict["modelVarIdxs"]	# list of lists containing indices of variables associated with each model
		
		# load ROM initial conditions from disk
		self.loadROMIC 		= catchInput(romDict, "loadROMIC", False)
		if self.loadROMIC:
			romIC 		= romDict["romIC"] 						# NumPy binary containing ROM initial conditions
			self.code0	= np.load(romIC, allow_pickle=True)		# list of NumPy arrays

			# check that IC dimensions match expected dimensions
			try:
				assert(len(self.code0) == self.numModels)
				for modelNum in self.numModels:
					assert(self.code0[modelNum].shape[0] == self.latentDims[modelNum])
			except:
				raise ValueError("ROM IC input dimensions did not match expected dimensions")

		# load linear basis for distributing to models
		if (self.romMethod == "linear"):
			linearBasis = np.load(os.path.join(self.modelDir, self.modelNames+".npy"))
			linearBasis = linearBasis[:, :, :max(self.latentDims)]

		# build solution
		# note: no copies, as this should map to same memory reference as sol
		self.solCons 	= sol.solCons
		self.solPrim 	= sol.solPrim
		self.RHS 		= sol.RHS 			
		
		# normalization/centering profiles
		self.normSubProf = self.loadStandardization(romDict["normSubIn"])
		self.normFacProf = self.loadStandardization(romDict["normFacIn"])
		self.centProf = self.loadStandardization(romDict["centIn"])

		# model/code associated with each decoder
		self.code 		= []
		self.modelList 	= []
		for modelID in range(self.numModels):
			self.code.append(np.zeros(self.latentDims[modelID]))
			if (self.romMethod == "linear"):
				self.modelList.append(model(modelID, self, linearBasis=linearBasis))
			elif (self.romMethod == "nonlinear"):
				self.modelList.append(model(modelID, self, encoderFlag=True))

		# nonlinear models are stored in separate files, linear basis is not
		try:
			assert(len(self.latentDims) == self.numModels)
		except:
			raise AssertionError("Incorrect number of latent dimension entries")

		try:
			if (self.romMethod == "nonlinear"):
				assert(len(self.modelNames) == self.numModels)
			elif (self.romMethod == "linear"):
				assert(type(self.modelNames) == str)
		except:
			raise AssertionError("Incorrect number/type of model names")


	def loadStandardization(self, dataIn):
		solShape = self.solPrim.shape
		try:
			if (type(dataIn) == list):
				assert(len(dataIn) == solShape[-1])
				standVals = np.array(dataIn, dtype=realType)		# load normalization subtraction values from user input
				standProf = np.ones(solShape, dtype=realType) * standVals
			elif (type(dataIn) == str):
				standProf = np.load(os.path.join(self.modelDir, dataIn))				# load normalization subtraction profile from file
				assert(standProf.shape == solShape)
		except:
			print("WARNING: standardization load failed or not specified, defaulting to zeros...")
			standProf = np.zeros(solShape, dtype=realType)

		return standProf

	# initialize code and solution, if required 
	def initializeROMState(self, sol: solutionPhys, params: parameters, gas: gasProps):

		if not self.loadROMIC:
			self.mapSolToModels(sol)

		for modelID in range(self.numModels):

			modelObj = self.modelList[modelID]

			# initialize from saved code
			if self.loadROMIC:
				modelObj.code = self.code0[modelId].copy()

			# compute projection of full-field initial condition
			else:
				if (params.timeScheme == "dualTime"):
					modelInput = modelObj.solPrim
				else:
					modelInput = modelObj.solCons
				modelInput = modelObj.standardizeData(modelInput)
				modelObj.code = modelObj.calcEncoding(modelInput)

			# make model inference, map to solution
			if (params.timeScheme == "dualTime"):
				modelObj.solPrim = modelObj.calcDecoding()
				sol.solPrim[:,modelObj.varIdxs] = modelObj.solPrim
			else:
				modelObj.solCons = modelObj.calcDecoding()
				sol.solCons[:,modelObj.varIdxs] = modelObj.solCons

		# finish up by initializing remainder of state
		if (params.timeScheme == "dualTime"):
			sol.updateState(gas)
		else:
			sol.updateState(gas, fromCons=True)

	# project RHS onto test space(s)
	def calcRHSProjection(self):

		for modelID in range(self.numModels):
			modelObj = self.modelList[modelID]
			modelObj.RHS = modelObj.standardizeData(modelObj.RHS, center=False)
			modelObj.RHSProj = modelObj.calcTestProjection(modelObj.RHS)


	# advance solution forward one subiteration
	def advanceSubiterExplicit(self, sol: solutionPhys, params: parameters, subiter, solOuter):

		for modelID in range(self.numModels):
			modelObj = self.modelList[modelID]
			modelObj.code = solOuter[modelID] + params.dt * params.subIterCoeffs[subiter] * modelObj.RHSProj
			# pdb.set_trace()
			modelObj.solCons = modelObj.calcDecoding(centering = True)
			sol.solCons[:,modelObj.varIdxs] = modelObj.solCons

	# simply returns a list containing the latent space solution
	def getCode(self):

		code = [None]*self.numModels
		for modelID in range(self.numModels):
			modelObj = self.modelList[modelID]
			code[modelID] = modelObj.code

		return code


	# distribute primitive and conservative fields to models
	def mapSolToModels(self, sol: solutionPhys):
		for modelID in range(self.numModels):
			modelObj = self.modelList[modelID]	
			modelObj.solCons = sol.solCons[:, modelObj.varIdxs]
			modelObj.solPrim = sol.solPrim[:, modelObj.varIdxs]


	# distribute RHS arrays to models for projection
	def mapRHSToModels(self, sol: solutionPhys):

		for modelID in range(self.numModels):
			modelObj = self.modelList[modelID]
			modelObj.RHS = sol.RHS[:, modelObj.varIdxs]

# encoder/decoder class
# TODO: model is a terrible name for this, need something that generalizes to linear/nonlinear and decoder/encoder
# TODO: some quantities are not required for the encoder (e.g. RHS), don't allocate unless necessary
class model:

	def __init__(self, modelID: int, rom: solutionROM, linearBasis=None, encoderFlag=False, consForm=True):

		self.modelID 	= modelID
		self.latentDim 	= rom.latentDims[modelID] 		# latent code size for model
		self.varIdxs 	= rom.modelVarIdxs[modelID]		# variable indices associated with model
		self.numVars 	= len(self.varIdxs)				# number of prim/cons variables associated with model
		self.numCells 	= rom.solCons.shape[0]			# number of cells in physical domain
		self.romMethod  = rom.romMethod 				# "linear" or "nonlinear"
		self.romProj 	= rom.romProj 					# "galerkin"
		self.consForm 	= consForm

		# encoder stuff
		self.encoderFlag = encoderFlag 												# whether or not the model has an encoder 
		if (self.romMethod == "nonlinear"): 
			self.encoderApprox = rom.encoderApprox 	# whether to use encoder projection approximation 
			self.modelJacob = np.zeros((self.numCells, self.numVars, self.latentDim), dtype=realType)
		else:
			self.encoderApprox = False

		# separate storage for input, output, and RHS 
		self.solCons 	= rom.solCons[:,self.varIdxs].copy()
		self.solPrim 	= rom.solPrim[:,self.varIdxs].copy()
		self.RHS 		= rom.RHS[:,self.varIdxs] 				# no copy for this, as this can just be a true reference
		self.solShape 	= self.solCons.shape
		self.code 		= np.zeros((self.latentDim,1), dtype=realType) # temporary space for any encodings/projections
		self.codeN 		= np.zeros((self.latentDim,1), dtype=realType) # code from last physical time step
		self.RHSProj 	= np.zeros((self.latentDim,1), dtype=realType) # projected RHS

		# normalization/centering arrays
		self.normSubProf 	= rom.normSubProf[:, self.varIdxs]
		self.normFacProf 	= rom.normFacProf[:, self.varIdxs]
		self.centProf 		= rom.centProf[:, self.varIdxs]

		# load linear basis and truncate modes
		# note: assumes that all bases (even scalar/separated) are stored as one concatenated basis array
		# could probably make this more efficient by loading it into solutionROM and distributing
		if (rom.romMethod == "linear"):
			# self.decoder 	= linearBasis[:,self.varIdxs,:self.latentDim]
			self.decoder 	= np.reshape(linearBasis[:,self.varIdxs,:self.latentDim], (-1, self.latentDim), order='C')

		# load TensorFlow model into decoder and encoder, assumed layer dimensions are in NCH format
		# TODO: add option to ingest full autoencoder and split, but I remember this being a pain in the ass
		elif (rom.romMethod == "nonlinear"):

			# load decoder
			modelLoc 		= os.path.join(rom.modelDir, "decoder_"+rom.modelNames[modelID]+".h5")
			self.decoder 	= load_model(modelLoc, compile=False)

			inputShape 		= self.checkIOShape(self.decoder.layers[0].input_shape)
			outputShape 	= self.checkIOShape(self.decoder.layers[-1].output_shape)
			try:
				assert(inputShape[-1] == self.latentDim)
			except:
				raise ValueError("Mismatched decoder input shapes: "+str(inputShape)+", "+str(self.latentDim))
			try:
				assert(outputShape[-1] == self.numCells)
				assert(outputShape[-2] == self.numVars)
			except:
				raise ValueError("Mismatched decoder output shapes: "+str(outputShape)+", "+str(self.solCons.shape))

			# load encoder
			modelLoc 		= os.path.join(rom.modelDir, "encoder_"+rom.modelNames[modelID]+".h5") 
			self.encoder 	= load_model(modelLoc, compile=False)

			inputShape 		= self.checkIOShape(self.encoder.layers[0].input_shape)
			outputShape 	= self.checkIOShape(self.encoder.layers[-1].output_shape)
			try:
				assert(inputShape[-1] == self.numCells)
				assert(inputShape[-2] == self.numVars)
			except:
				raise ValueError("Mismatched encoder input shapes: "+str(inputShape)+", "+str(self.solCons.shape))
			try:
				assert(outputShape[-1] == self.latentDim)
			except:
				raise ValueError("Mismatched encoder output shapes: "+str(outputShape)+", "+str(self.latentDim))

		# storage for projection matrix
		if ((rom.romMethod == "linear") and (rom.romProj == "galerkin")):
			# self.projector = np.reshape(np.transpose(self.decoder, axes=(2,0,1)), (-1,self.numCells*self.numVars), order='C')
			self.projector = self.decoder.T
		else:
			self.projector = np.zeros((self.latentDim, self.numCells*self.numVars), dtype = realType)

	# check model input/output shape format
	# should be a tuple. If list, check that it only has one element and convert 
	def checkIOShape(self, shape):

		if type(shape) is list:
			if (len(shape) != 1):
				raise ValueError("Invalid TF model I/O size")
			else:
				shape = shape[0]

		return shape


	# run decoder, de-normalize, and de-center
	# TODO: add casting to same datatype as network
	def calcDecoding(self, centering=True):

		solDecode = np.zeros(self.solShape, dtype=realType)

		if (self.romMethod == "linear"):
			solDecode = self.decoder @ self.code 
			solDecode = np.reshape(solDecode, (self.numCells, self.numVars), order='C')

		elif (self.romMethod == "nonlinear"):
			# input expected to be in NW format
			# output is in NCW format
			# pdb.set_trace()
			solDecode = np.squeeze(self.decoder.predict(self.code[None,:])).T

		# de-normalize and de-center
		solDecode = self.standardizeData(solDecode, centering, inverse=True)

		return solDecode

	# run "encoder" 
	# for linear this is just Galerkin projection, for NLM it's running through the encoder
	# assumes input has already been scaled
	# TODO: maybe add scaling to this
	# TODO: add casting to same datatype as network
	def calcEncoding(self, encodeVec):

		# Galerkin projection
		if (self.romMethod == "linear"):
			encoding = self.projector @ encodeVec.flatten(order='C')

		# nonlinear encoder
		else:
			# model input expected to be in NCW format
			# model output is in NW format
			encoding = np.squeeze(self.encoder.predict(encodeVec.T[None,:,:]).T)

		return encoding

	# project a vector (presumably solution or RHS vector) onto test space
	def calcTestProjection(self, projVec):

		proj = np.zeros(self.latentDim, dtype=realType)
		if (self.romMethod == "linear"):
			if (self.romProj == "galerkin"):
				pass # this is constant
			else:
				raise ValueError("Non-Galerkin linear projection not yet implemented")


		elif (self.romMethod == "nonlinear"):
			if self.encoderApprox:
				raise ValueError("Encoder projection not yet implemented")
			else:
				calcAnalyticalTFJacobian(self) # compute decoder Jacobian
				self.projector = pinv(np.reshape(self.modelJacob, (-1, self.latentDim), order='C'))

		# compute projection onto test space
		proj = self.projector @ projVec.flatten(order='C')

		# pdb.set_trace()

		return proj


	# (de)centering and (de)normalization
	def standardizeData(self, solArr, center=True, inverse=False):

		if inverse:
			solArr = self.normalizeSol(solArr, denormalize=True)
			if center: solArr = self.centerSol(solArr, decenter=True)
		else:
			if center: solArr = self.centerSol(solArr, decenter=False)
			solArr = self.normalizeSol(solArr, denormalize=False)
		
		return solArr 


	# center/decenter full-order solution
	def centerSol(self, solArr, decenter = False):

		if decenter:
			solArr += self.centProf
		else:
			solArr -= self.centProf
		return solArr


	# normalize/denormalize full-order solution
	def normalizeSol(self, solArr, denormalize = False):

		if denormalize:
			solArr = solArr * self.normFacProf + self.normSubProf
		else:
			solArr = (solArr - self.normSubProf) / self.normFacProf
		return solArr


	