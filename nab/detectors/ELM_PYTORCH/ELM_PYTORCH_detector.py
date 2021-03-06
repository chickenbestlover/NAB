
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from nab.detectors.base import AnomalyDetector
from nupic.algorithms import anomaly_likelihood
import math
import random
from model import ELMModel
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
from pseudoInverse import pseudoInverse
# Fraction outside of the range of values seen so far that will be considered
# a spatial anomaly regardless of the anomaly likelihood calculation. This
# accounts for the human labelling bias for spatial values larger than what
# has been seen so far.
SPATIAL_TOLERANCE = 0.05



class ELM_PYTORCHDetector(AnomalyDetector):
  #def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction,BN=True,AE=True,ORTH=True,
  #             inputWeightForgettingFactor=0.999,
  #            outputWeightForgettingFactor=0.999,
  #             hiddenWeightForgettingFactor=0.999):
  def __init__(self, *args, **kwargs):
    super(ELM_PYTORCHDetector, self).__init__(*args, **kwargs)
    random.seed(6)
    self.cuda = True
    self.mean = 0.0
    self.squareMean = 0.0
    self.std = 0.0
    self.windowSize = 1000
    self.pastData = [0.0]
    self.inputCount = 0
    self.minVal = None
    self.maxVal = None
    self.anomalyLikelihood = None
    # Set this to False if you want to get results based on raw scores
    # without using AnomalyLikelihood. This will give worse results, but
    # useful for checking the efficacy of AnomalyLikelihood. You will need
    # to re-optimize the thresholds when running with this setting.
    self.useLikelihood = True
    if self.useLikelihood:
      # Initialize the anomaly likelihood object
      numentaLearningPeriod = int(math.floor(self.probationaryPeriod / 2.0))
      self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        claLearningPeriod=numentaLearningPeriod,
        estimationSamples=self.probationaryPeriod-numentaLearningPeriod,
        reestimationPeriod=100
      )

    self.inputSequenceWindowLen = 30
    self.predictionStep = self.inputSequenceWindowLen

    self.predValues= [0.0] * self.predictionStep
    self.prevPredValues = [0.0] * self.predictionStep
    self.predictionErrorsInWindow = torch.FloatTensor(1,self.predictionStep).zero_()
    if self.cuda:
      self.predictionErrorsInWindow = self.predictionErrorsInWindow.cuda()


    self.totalSequence = [0.0] * (self.inputSequenceWindowLen + self.predictionStep)
    self.inputSequence = [0.0] * self.inputSequenceWindowLen
    self.targetSequence = [0.0] * self.predictionStep

    self.pastTestOutputSequences = [0.0]*self.predictionStep


    self.input_size = self.inputSequenceWindowLen
    self.hidden_size = 200
    self.output_size = self.predictionStep
    self.model = ELMModel(input_size=self.input_size,output_size=self.output_size,
                          hidden_size=self.hidden_size,layerNorm=True, forgettingFactor=1)

    self.batch_size=1
    if self.cuda:
      self.model.cuda()
    self.criterion = nn.MSELoss()
    self.optimizer = pseudoInverse(params=self.model.parameters(),C=1e-2)
    self.FIRSTBATCH=True

  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["predValueFirst","predValueLast"]

  def updateLearningSequences(self, newInput):

    self.totalSequence.append(newInput)
    self.totalSequence.pop(0)

    self.inputSequence = self.totalSequence[:self.inputSequenceWindowLen]
    self.targetSequence = self.totalSequence[self.inputSequenceWindowLen:]

  def updatePastData(self, newInput):
    self.inputCount = self.inputCount+1
    self.pastData.append(newInput)
    if len(self.pastData)>self.windowSize:
      self.pastData.pop(0)

    self.mean = np.mean(self.pastData)
    self.std = np.std(self.pastData)
    #self.std =   torch.var(torch.FloatTensor(self.pastData))


  def normalize(self, input):

    return (input-self.mean)/(self.std + 0.00001)

  def reconstruct(self, input):

    return input*self.std + self.mean

  def getInputSequenceBatchAsTensor(self):

    nInputSeq = torch.FloatTensor(self.inputSequence).view(1, len(self.inputSequence))
    nInputSeqBatch = nInputSeq
    for i in range(1,self.batch_size):
      nInputSeqBatch = torch.cat((nInputSeqBatch,nInputSeq+0.7*(i/self.batch_size)*torch.randn(nInputSeq.size())),0)
    if self.cuda:
      nInputSeqBatch = nInputSeqBatch.cuda()

    return nInputSeqBatch

  def getTargetSequenceBatchAsTensor(self):

    nTargetSeq = torch.FloatTensor(self.targetSequence).view(1,len(self.targetSequence))
    nTargetSeqBatch = nTargetSeq
    for i in range(1,self.batch_size):
      nTargetSeqBatch = torch.cat((nTargetSeqBatch,nTargetSeq+0.7*(i/self.batch_size)*torch.randn(nTargetSeq.size())),0)
    if self.cuda:
      nTargetSeqBatch =nTargetSeqBatch.cuda()

    return nTargetSeqBatch

  def getTestSequenceAsTensor(self):

    nTestSeq = torch.FloatTensor(self.targetSequence).view(1,len(self.targetSequence))
    if self.cuda:
      nTestSeq =nTestSeq.cuda()

    return nTestSeq

  def updatePastTestOutputSequences(self, nOutputSeq):
    self.pastTestOutputSequences.append(nOutputSeq)
    return self.pastTestOutputSequences.pop(0)

  def updatePredictionErrors(self,nPredictionErrors):
    if self.inputCount<self.windowSize:
      self.predictionErrorsInWindow = torch.cat((self.predictionErrorsInWindow,nPredictionErrors),0)
    else:
      self.predictionErrorsInWindow = torch.cat((self.predictionErrorsInWindow[1:],nPredictionErrors),0)


  def calculateCov(self,data):

    data = data.t()
    mean_data = data.mean(1)
    xm = data.sub(mean_data.expand_as(data))
    cov = xm.mm(xm.t())
    cov = cov/(data.size(1))
    return cov

  def train_firstBatch(self, nInputSeqBatch, nTargetSeqBatch):
    nHiddenOutBatch= self.model.forwardToHidden(nInputSeqBatch)
    self.optimizer.train(nHiddenOutBatch,nTargetSeqBatch)

  def train_sequential(self, nInputSeqBatch, nTargetSeqBatch):
    nHiddenOutBatch= self.model.forwardToHidden(nInputSeqBatch)
    self.optimizer.train_sequential(nHiddenOutBatch,nTargetSeqBatch)


  def predict(self, nInput):
    nOutput = self.model.forward(nInput)

    return nOutput



  def computeRawAnomaly(self, trueVal, predVal,saturation=True):

    #AbsolutePercentageError = np.abs(trueVal- predVal) / (np.abs(trueVal)+0.00001)
    pastTrueVal=self.inputSequence[len(self.inputSequence)-1]
    AbsolutePercentageError = np.abs(trueVal- predVal) / (2*np.abs(trueVal-pastTrueVal)+0.00001)
    if saturation:
      AbsolutePercentageError = min(1.0,AbsolutePercentageError)



    return AbsolutePercentageError

  def handleRecord(self, inputData):


    # Get the value
    value = inputData["value"] # value: float
    self.updatePastData(value) # re-calcucate mean, var for past 1000 samples, and update self.inputCount
    self.prevPredValues=self.predValues

#    print 'prevPredValue = {:6.4f}   |   trueValue = {:6.4f}'.format(self.prevPredValues[-1],value)
    nPrevPredValues = [self.normalize(prevPredValue) for prevPredValue in self.prevPredValues]
    nValue = self.normalize(value)
    self.updateLearningSequences(nValue)
    '''
    Step1: Anomaly detection using previous prediction at (t_now - 1) and current input value at (t_now)
    '''
    nPredictions = [0.0]*self.predictionStep
    nPredictionErrors = [0.0]*self.predictionStep
    #print self.inputCount

    if self.inputCount > self.predictionStep:

      nPredictions = [self.pastTestOutputSequences[i].cpu().data[0][-1-i] for i in range(self.predictionStep)]
      nPredictionErrors = [ nPrediction - nValue for nPrediction in nPredictions]
      nPredictionErrors = torch.FloatTensor(nPredictionErrors).view(1, len(nPredictionErrors))
      if self.cuda:
        nPredictionErrors=nPredictionErrors.cuda()
      if len(self.predictionErrorsInWindow) > self.predictionStep:
        multivariateGaussMean = self.predictionErrorsInWindow.mean(0)
        multivariateGaussCov = self.calculateCov(self.predictionErrorsInWindow)
        determinant = np.linalg.det(multivariateGaussCov.cpu().numpy())
        #0#0000000d00ob[0][0]

      self.updatePredictionErrors(nPredictionErrors)
      #print self.predictionErrorsInWindow
      #print multivariateGaussCov.size()
      #print nPredictionsErrors

    finalScore=0.0
    if self.inputCount>self.hidden_size:
      rawScore = self.computeRawAnomaly(trueVal=nValue, predVal=nPredictions[-1], saturation=True)

      if self.useLikelihood:
        # Compute log(anomaly likelihood)
        anomalyScore = self.anomalyLikelihood.anomalyProbability(
         inputData["value"], abs(nPredictionErrors.cpu()[0].mean()), inputData["timestamp"])
   #       inputData["value"], rawScore, inputData["timestamp"])

        logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
        finalScore = logScore
        if self.inputCount<self.hidden_size+100:
          finalScore=0.0
        # print finalScore
      else:
        finalScore = rawScore

      # Update min/max values and check if there is a spatial anomaly
      spatialAnomaly = False
      if self.minVal != self.maxVal:
        tolerance = (self.maxVal - self.minVal) * SPATIAL_TOLERANCE
        maxExpected = self.maxVal + tolerance
        minExpected = self.minVal - tolerance
        if value > maxExpected or value < minExpected:
          spatialAnomaly = True
      if self.maxVal is None or value > self.maxVal:
        self.maxVal = value
      if self.minVal is None or value < self.minVal:
        self.minVal = value
      if spatialAnomaly:
        finalScore = 1.0




    '''
    Step2: Network training
    '''
    # nInputSeqBatch: FloatTensor(batch_size,input_size)
    nInputSeqBatch = Variable(self.getInputSequenceBatchAsTensor(),volatile=True)
    nTargetSeqBatch = Variable(self.getTargetSequenceBatchAsTensor(),volatile=True)

    if self.inputCount < self.hidden_size:
      pass
    elif self.FIRSTBATCH:

      nHidOutputSeqBatch = self.model.forwardToHidden(nInputSeqBatch)
      self.optimizer.train(nHidOutputSeqBatch,nTargetSeqBatch)
      self.FIRSTBATCH=False

    else:
      nHidOutputSeqBatch = self.model.forwardToHidden(nInputSeqBatch)
      self.optimizer.train_sequential(nHidOutputSeqBatch,nTargetSeqBatch)

    #print "train MSE loss = {:6.6f}".format(trainLoss)

    #print Variable(nTargetSeqBatch)[-1][0].view(1,1,1)
    #nPredValue = self.predict(nTargetSeqBatch[-1][0].view(1,1,1))

    nTestSeq = Variable(self.getTestSequenceAsTensor(),volatile=True)


    nOutputSeq = self.predict(nTestSeq)



    nPastOutputSeq = self.updatePastTestOutputSequences(nOutputSeq)
    #print nPastOutputSeq
    nPastPred=0.0
    if self.inputCount > self.predictionStep:
      testLoss = self.criterion(nTargetSeqBatch[0].squeeze(0),nPastOutputSeq)
      #print "test MSE loss  = {:6.6f}".format(testLoss.cpu().data[0])
      nPastPred = self.reconstruct(nPastOutputSeq.cpu().data[0][-1])
      #print pastPred
    nPredValues= [ nOutputSeq.cpu().data[0][i] for i in range(self.predictionStep) ]


    #print nPredValues

    self.predValues = [self.reconstruct(nPredValue) for nPredValue in nPredValues]

    #del nInputSeqBatch
    #del nTargetSeqBatch

    #return (finalScore, np.mean(self.prevPredValues))
    return (finalScore, self.prevPredValues[0], 0.0)

