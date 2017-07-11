
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from nab.detectors.base import AnomalyDetector
from nupic.algorithms import anomaly_likelihood
import math
import random
from model import RNNModel
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
# Fraction outside of the range of values seen so far that will be considered
# a spatial anomaly regardless of the anomaly likelihood calculation. This
# accounts for the human labelling bias for spatial values larger than what
# has been seen so far.
SPATIAL_TOLERANCE = 0.05



class LSTMDetector(AnomalyDetector):
  #def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction,BN=True,AE=True,ORTH=True,
  #             inputWeightForgettingFactor=0.999,
  #            outputWeightForgettingFactor=0.999,
  #             hiddenWeightForgettingFactor=0.999):
  def __init__(self, *args, **kwargs):
    super(LSTMDetector, self).__init__(*args, **kwargs)
    random.seed(6)
    self.mean = 0.0
    self.squareMean = 0.0
    self.var = 0.0
    self.windowSize = 1000
    self.pastData = [0.0] * self.windowSize
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
    self.inputSequenceLen_forBPTT = 1

    self.predValue=0
    self.prevPredValue=0
    self.inputSequence = [0.0] * self.inputSequenceLen_forBPTT
    self.cuda = True
    self.model = RNNModel(rnn_type='GRU',input_size=1,
                          embed_size=25,hidden_size=25,
                          nlayers=1,dropout=0.0,tie_weights=False)
    self.batch_size=10
    if self.cuda:
      self.model.cuda()
    self.hidden_for_train = self.model.init_hidden(bsz=self.batch_size)
    self.hidden_for_test = self.model.init_hidden(bsz=1)
    self.criterion = nn.MSELoss()
    #self.criterion = nn.L1Loss()
    #self.criterion = nn.SmoothL1Loss()
    self.clip = 0.5
    self.optimizer = optim.Adam(self.model.parameters(),lr=0.05)
    #self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.01)

  def getAdditionalHeaders(self):
    """Returns a list of strings."""
    return ["predValue"]

  def updateInputSequence(self, newInput):

    self.inputSequence.append(newInput)
    self.inputSequence.pop(0)

  def updatePastData(self, newInput):
    self.inputCount = self.inputCount+1
    if self.inputCount==1:
      self.mean = self.mean + newInput / (self.inputCount)
      self.squareMean = self.squareMean + newInput * newInput / self.inputCount
      self.var = 0.0

    elif self.inputCount< self.windowSize:
      self.mean = self.mean + newInput / (self.inputCount) - self.pastData[0] / (self.inputCount)
      self.squareMean = self.squareMean + newInput * newInput / self.inputCount - self.pastData[0] * self.pastData[
        0] / self.inputCount
      self.var = (self.squareMean - self.mean * self.mean / self.inputCount) / (self.inputCount - 1)
    else :
      self.mean = self.mean + newInput/(self.windowSize) - self.pastData[0]/(self.windowSize)
      self.squareMean = self.squareMean + newInput*newInput/self.windowSize - self.pastData[0]*self.pastData[0]/self.windowSize
      self.var = (self.squareMean - self.mean*self.mean/self.windowSize)/(self.windowSize-1)

    self.pastData.append(newInput)
    self.pastData.pop(0)

  def normalize(self, input):

    return (input-self.mean)/(self.var+0.00001)

  def reconstruct(self, input):

    return input*self.var + self.mean

  def getInputSequenceBatchAsTensor(self):

    nInputSeq = torch.FloatTensor(self.inputSequence).view(self.inputSequenceLen_forBPTT,1,1)
    nInputSeqBatch = nInputSeq
    for i in range(1,self.batch_size):
      nInputSeqBatch = torch.cat((nInputSeqBatch,nInputSeq+0.5*(i/self.batch_size)*torch.randn(nInputSeq.size())),1)
    if self.cuda:
      nInputSeqBatch = nInputSeqBatch.cuda()
    #print nInputSeqBatch
    #raw_input('hello')

    return nInputSeqBatch

  def train(self, nInputSeqBatch_forBPTT, nTargetSeqBatch_forBPTT):
    self.model.train()
    self.hidden_for_train = self.repackage_hidden(self.hidden_for_train)
    #print self.hidden_for_train[0].size()
    #print self.hidden_for_train[1].size()
    self.model.zero_grad()
    nOutputSeqBatch_forBPTT, self.hidden_for_train = self.model.forward(nInputSeqBatch_forBPTT,self.hidden_for_train)
    #print nOutputSeqBatch_forBPTT.size()
    loss = self.criterion(nOutputSeqBatch_forBPTT.view(-1,1),nTargetSeqBatch_forBPTT)
    loss.backward()
    torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clip)
    self.optimizer.step()
    #print loss.cpu().data[0]

    return loss.cpu().data[0]

  def predict(self, nInput):
    self.model.eval()
    self.model.init_hidden(bsz=1)
    nOutput, self.hidden_for_test = self.model.forward(nInput,self.hidden_for_test)

    return nOutput.cpu().data[0][0][0]

  def repackage_hidden(self,hidden):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(hidden) == Variable:
      return Variable(hidden.data)
    else:
      return tuple(self.repackage_hidden(v) for v in hidden)


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
    self.prevPredValue=self.predValue
    nPrevPredValue = self.normalize(self.prevPredValue)
    nValue = self.normalize(value)
    '''
    Step1: Anomaly detection using previous prediction at (t_now - 1) and current input value at (t_now)
    '''

    finalScore=0.0
    if self.inputCount > 100:
      rawScore = self.computeRawAnomaly(trueVal=nValue, predVal=nPrevPredValue, saturation=True)

      if self.useLikelihood:
        # Compute log(anomaly likelihood)
        anomalyScore = self.anomalyLikelihood.anomalyProbability(
          inputData["value"], rawScore, inputData["timestamp"])
        logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
        finalScore = logScore
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

    if self.inputCount>500:
      # nInputSequenceBatch_forBPTT: FloatTensor(seq_len,batch_size,input_size)
      nInputSeqBatch_forBPTT = self.getInputSequenceBatchAsTensor()
      #if self.inputCount<50:
      #  print self.inputCount, inputData['value'], nValue, nInputSeqBatch_forBPTT[:,0,0]

      self.updateInputSequence(nValue)
      nTargetSeqBatch_forBPTT = self.getInputSequenceBatchAsTensor()
      if self.inputCount % self.inputSequenceLen_forBPTT == 0:

        #print '\n', self.inputCount
        #print nInputSeqBatch_forBPTT.size()
        #print nTargetSeqBatch_forBPTT.size()
        #print self.model
        self.train(Variable(nInputSeqBatch_forBPTT),Variable(nTargetSeqBatch_forBPTT))

      #print Variable(nTargetSeqBatch_forBPTT)[-1][0].view(1,1,1)
      nPredValue = self.predict(Variable(nTargetSeqBatch_forBPTT)[-1][0].view(1,1,1))
      self.predValue = self.reconstruct(nPredValue)

      del nInputSeqBatch_forBPTT
      del nTargetSeqBatch_forBPTT

    return (finalScore, self.prevPredValue)

