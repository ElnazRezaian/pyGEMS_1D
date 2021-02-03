import numpy as np

class adapt():
    def __init__(self, model, solver, romDomain):
        self.adaptionMethod = romDomain.adaptionMethod
        self.subIteration   = False
        if self.adaptionMethod == "OSAB":
            """Unpacking parameters for One-Step adaptive Basis method
            parameters required = [#, #, ...]
            DataType = [int, ]
            
            
            """
            parameterRequirement = 1
            self.residualSamplingStep = romDomain.adaptionParameters[0]   ## After # time steps residual will be sampled
            assert(len(romDomain.adaptionParameters) == parameterRequirement), "Insufficient/over-specified adaption input parameters"
            self.subIteration = True
            self.trueStandardizedState    = np.zeros((model.numVars, solver.mesh.numCells))
            self.adaptionResidual         = np.zeros((model.numVars*solver.mesh.numCells))
            self.basisUpdate              = np.zeros(model.trialBasis.shape)


        else:
            raise ValueError ("Invalid selection for model adaptation method")

    def adaptModel(self, romDomain, solDomain, solver, model):
        if romDomain.adaptionMethod == "OSAB":
            self.adaptionResidual = (self.trueStandardizedState - model.applyTrialBasis(model.code)).flatten(order = "C").reshape(-1, 1)
            self.basisUpdate = np.dot(self.adaptionResidual, model.code.reshape(1, -1)) / np.linalg.norm(model.code)**2
            model.trialBasis = model.trialBasis + self.basisUpdate
            model.updateSol(solDomain)
            solDomain.solInt.updateState(solver.gasModel, fromCons=True)












