import os
import numpy as np
from inputFuncs import readInputFile
from classDefs import parameters, catchInput
from solution import solutionPhys
import constants
from constants import realType
# import tensorflow as tf 
# from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pdb

# TODO: a little too liberal with similar variable names between solutionROM and model
# TODO: quite a bit changes if using primitive variables or conservative variables
# 		This includes VTROM, and which variables are projected

# overarching class containing all info/objects/methods required to compute ROMs
class solutionROM:

	def __init__(self, romFile, sol: solutionPhys, params: parameters):

		# read input parameters
		romDict = readInputFile(romFile)

		# ROM methods
		self.romMethod 		= catchInput(romDict, "romMethod", "linear") 		# accepts "linear" or "nonlinear"
		if (self.romMethod == "nonlinear"):
			self.encoderApprox = catchInput(romDict, "encoderApprox", False)	# whether to make encoder projection approximation
		else:
			self.encoderApprox = False
		self.romProj 		= catchInput(romDict, "romProj", "galerkin") 		# accepts "galerkin" or "lspg"
		params.simType 		= self.romMethod + "_" + self.romProj

		# model parameters
		self.modelDir 		= romDict["modelDir"]		# location of all models
		self.numModels 		= romDict["numModels"]		# number of models (trial bases/decoders) being used
		self.modelNames 	= romDict["modelNames"] 	
		self.latentDims 	= romDict["latentDims"]		# list of latent code dimensions for each model
		self.modelVarIdxs 	= romDict["modelVarIdxs"]	# list of lists containing indices of variables associated with each model
		
		# load ROM initial conditions from disk
		self.loadROMIC 		= catchInput(romDict, "loadROMIC", False)
		if self.loadROMIC:
			romIC 		= romDict["romIC"] 	# NumPy binary containing ROM initial conditions
			self.code0	= np.load(romIC)
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
		self.code 			= []
		self.decoderList 	= []
		self.encoderList 	= []
		for modelID in range(self.numModels):
			self.code.append(np.zeros(self.latentDims[modelID]))
			if (self.romMethod == "linear"):
				self.decoderList.append(model(modelID, self, linearBasis=linearBasis))
			elif (self.romMethod == "nonlinear"):
				self.decoderList.append(model(modelID, self))
				if self.encoderApprox:
					self.encoderList.append(model(modelID, self, encoderFlag=True))

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
			print("WARNING: normSubIn load failed or not specified, defaulting to zeros...")
			standProf = np.zeros(solShape, dtype=realType)

		return standProf

	# initialize code and solution, if required 
	def initializeROMState(self, sol: solutionPhys):

		if not self.loadROMIC:
			self.mapSolToModels(sol)

		for modelID in range(self.numModels):

			# just compute Galerkin projection of solution
			if (self.romMethod == "linear"):
				
				decoder = self.decoderList[modelID]
				
				# initialize from saved code
				if self.loadROMIC:
					decoder.code = self.code0.copy()	
				# project onto test space
				else:
					decoder.solCons = decoder.standardizeData(decoder.solCons)
					decoder.code = decoder.calcProjection(decoder.solCons)

				decoder.solCons = decoder.inferModel()
				sol.solCons[:,decoder.varIdxs] = decoder.solCons

			elif (self.romMethod == "nonlinear"):
				raise ValueError("Nonlinear initialization not yet implemented")

	# project RHS onto test space(s)
	def calcRHSProjection(self):

		for modelID in range(self.numModels):
			if self.encoderApprox:
				raise ValueError("Encoder projection not yet implemented")
			else:
				modelObj = self.decoderList[modelID]

			# pdb.set_trace()
			modelObj.RHS = modelObj.standardizeData(modelObj.RHS, center=False)
			# pdb.set_trace()
			modelObj.RHSProj = modelObj.calcProjection(modelObj.RHS)
			# pdb.set_trace()


	# advance solution forward one subiteration
	# TODO: right now returning dSolCons, really should just return next time step but requires input of previous time step solution
	def advanceSubiter(self, sol: solutionPhys, params: parameters, subiter, solOuter):

		# if linear, can just compute change in code, don't need to decenter
		# dSolCons = np.zeros(self.solCons.shape, dtype=realType)
		for modelID in range(self.numModels):
			decoder = self.decoderList[modelID]

			if (self.romMethod == "linear"):
				# dCode = params.dt * params.subIterCoeffs[subiter] * decoder.RHSProj
				decoder.code = solOuter[modelID] + params.dt * params.subIterCoeffs[subiter] * decoder.RHSProj
				decoder.solCons = decoder.inferModel(centering = True)
				sol.solCons[:,decoder.varIdxs] = decoder.solCons
				
			# for nonlinear, need to compute next code step explicitly, decode
			elif (self.romMethod == "nonlinear"):
				raise ValueError("Nonlinear subiter advance not yet implemented")


	# simply returns a list containing the latent space solution
	def getCode(self):

		code = [None]*self.numModels
		for modelID in range(self.numModels):
			decoder = self.decoderList[modelID]
			code[modelID] = decoder.code

		return code

	# # collect decoder inferences into sol
	# # note: needs to map to solPrim or solCons, depending on how the model is trained
	# def mapDecoderToSol(self):


	# # collect encoder inferences into code
	# def mapEncoderToCode(self):

	# distribute primitive and conservative fields to models
	def mapSolToModels(self, sol: solutionPhys):
		for modelID in range(self.numModels):
			if (self.romMethod == "linear"):
				modelObj = self.decoderList[modelID]
			
			elif (self.romMethod == "nonlinear"): 
				raise ValueError("Nonlinear ROM not yet implemented")
				if self.encoderApprox:
					raise ValueError("Encoder projection not yet implemented")
				
			modelObj.solCons = sol.solCons[:, modelObj.varIdxs]
			modelObj.solPrim = sol.solPrim[:, modelObj.varIdxs]


	# distribute RHS arrays to models for projection
	def mapRHSToModels(self, sol: solutionPhys):

		for modelID in range(self.numModels):
			if self.encoderApprox:
				raise ValueError("Encoder projection not yet implemented")
			else:
				modelObj = self.decoderList[modelID]
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
		self.encoderFlag = encoderFlag 						# whether or not the model is an encoder
		self.romMethod  = rom.romMethod 				# "linear" or "nonlinear"
		self.romProj 	= rom.romProj 					# "galerkin"
		self.consForm 	= consForm

		# separate storage for input, output, and RHS 
		self.solCons 	= rom.solCons[:,self.varIdxs].copy()
		self.solPrim 	= rom.solPrim[:,self.varIdxs].copy()
		self.RHS 		= rom.RHS[:,self.varIdxs] 				# no copy for this, as this can just be a true reference
		self.solShape 	= self.solCons.shape
		self.code 		= np.zeros((self.latentDim,1), dtype=realType) # temporary space for any encodings/projections
		self.codeN 		= np.zeros((self.latentDim,1), dtype=realType) # code from last physical time step
		self.RHSProj 	= np.zeros((self.latentDim,1), dtype=realType) # projected RHS

		# load encoder/decoder
		# if (rom.romMethod == linear):
		# 	modelFile = os.path.join(rom.modelDir, "lin"

		# normalization/centering arrays
		self.normSubProf 	= rom.normSubProf[:, self.varIdxs]
		self.normFacProf 	= rom.normFacProf[:, self.varIdxs]
		self.centProf 		= rom.centProf[:, self.varIdxs]

		# storage for projection matrix
		self.projector 		= np.zeros((self.latentDim, self.numCells, self.numVars), dtype = realType)


		# load linear basis and truncate modes
		# note: assumes that all bases (even scalar/separated) are stored as one concatenated basis array
		# could probably make this more efficient by loading it into solutionROM and distributing
		if (rom.romMethod == "linear"):
			self.decoder 	= linearBasis[:,self.varIdxs,:self.latentDim]

		# load TensorFlow model into encoder or decoder, assumed layer dimensions are in NCH format
		# note: must be split separately as a decoder or encoder model
		# TODO: add option to ingest full autoencoder and split, but I remember this being a pain in the ass
		elif (rom.romMethod == "nonlinear"):

			if encoderFlag:
				modelLoc 		= os.path.join(rom.modelDir, "encoder_"+rom.modelNames[modelID]+".h5") 
				self.encoder 	= load_model(modelLoc)
				inputShape 		= self.encoder.layers[0].input_shape
				outputShape 	= self.encoder.layers[-1].output_shape
				
				try:
					assert(inputShape[-1] == self.numCells)
					assert(inputShape[-2] == self.numVars)
				except:
					raise ValueError("Mismatched encoder input shapes: "+inputShape+", "+str(self.solCons.shape))
				try:
					assert(outputShape[-1] == self.latentDim)
				except:
					raise ValueError("Mismatched encoder output shapes: "+outputShape+", "+str(self.latentDim))

			else:
				modelLoc 		= os.path.join(rom.modelDir, "decoder_"+rom.modelNames[modelID]+".h5")
				self.decoder 	= load_model(modelLoc)
				inputShape 		= self.decoder.layers[0].input_shape
				outputShape 	= self.decoder.layers[-1].output_shape
				try:
					assert(inputShape[-1] == self.latentDim)
				except:
					raise ValueError("Mismatched decoder input shapes: "+inputShape+", "+str(self.latentDim))

				try:
					assert(outputShape[-1] == self.numCells)
					assert(outputShape[-2] == self.numVars)
				except:
					raise ValueError("Mismatched decoder output shapes: "+outputShape+", "+str(self.solCons.shape))

	# run model inference
	def inferModel(self, centering=True):

		# if encoder, center and normalize, run evaluation
		if self.encoderFlag:
			raise ValueError("Encoder inference not yet implemented")

		# if decoder, run evaluation, denormalize and decenter sol
		else:
			solDecode = np.zeros(self.solShape, dtype=realType)

			if (self.romMethod == "linear"):
				for varIdx in range(self.numVars):
					solDecode[:, varIdx] = self.decoder[:,varIdx,:] @ self.code
					# pdb.set_trace()

			elif (self.romMethod == "nonlinear"):
				raise ValueError("Nonlinear manifold decoder not yet implemented")

			solDecode = self.standardizeData(solDecode, centering, inverse=True)

			return solDecode


	# project a vector (presumably solution or RHS vector) onto latent space
	def calcProjection(self, projVec):

		if self.encoderFlag:
			raise ValueError("Encoder projection not yet implemented")

		else:

			# TODO: only do this once if it's static linear galerkin, don't need to keep reassigning it
			if (self.romMethod == "linear"):
				self.calcLinearProjector()

			elif (self.romMethod == "nonlinear"):
				raise ValueError("Nonlinear decoder projection not yet implemented")

		# compute projection
		proj = np.zeros(self.latentDim, dtype=realType)
		for varIdx in range(self.numVars):
			proj += self.projector[:,:,varIdx] @ projVec[:,varIdx]

		return proj

	# compute projection matrix for linear model
	def calcLinearProjector(self):

		if (self.romProj == "galerkin"):
			self.projector = np.transpose(self.decoder, axes=(2,0,1))

		else:
			raise ValueError("Invalid choice of projection method: "+self.romProj)

	# # compute projection matrix for nonlinear model
	# def calcNonlinearProjector(self):

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


	