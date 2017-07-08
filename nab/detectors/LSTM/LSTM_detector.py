
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from nab.detectors.base import AnomalyDetector
from nupic.algorithms import anomaly_likelihood
import math
import random
from model import RNNModel
import torch
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
      numentaLearningPeriod = math.floor(self.probationaryPeriod / 2.0)
      #print 'numentaLearningPeriod=',numentaLearningPeriod
      self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        learningPeriod=numentaLearningPeriod,
        estimationSamples=self.probationaryPeriod - numentaLearningPeriod,
        reestimationPeriod=100
      )
    self.inputSequenceLen_forBPTT = 50
    self.outputs = 1
    self.numHiddenNeurons = 15
    self.predValue=0
    self.prevPredValue=0
    self.inputSequence = [0.0] * self.inputSequenceLen_forBPTT
    self.cuda = True
    self.model = RNNModel(rnn_type='LSTM',input_size=1,
                          embed_size=200,hidden_size=200,
                          nlayers=2,dropout=0.5,tie_weights=False)
    self.batch_size=20
    if self.cuda:
      self.model.cuda()

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
      nInputSeqBatch = torch.cat((nInputSeqBatch,nInputSeq+(i/self.batch_size)*torch.randn(nInputSeq.size())),1)
    if self.cuda:
      nInputSeqBatch = nInputSeqBatch.cuda()
    #print nInputSeqBatch
    #raw_input('hello')

    return nInputSeqBatch

  def train(self, features, targets):
    pass

  def predict(self, features):
    """
    Make prediction with feature matrix
    :param features: feature matrix with dimension (numSamples, numInputs)
    :return: predictions with dimension (numSamples, numOutputs)
    """
    #print features.flatten()[-1]
    prediction = np.array([[features.flatten()[-1]]])
    return prediction

  def computeRawAnomaly(self, trueVal, predVal,saturation=True):

    #AbsolutePercentageError = np.abs(trueVal- predVal) / (np.abs(trueVal)+0.00001)
    pastTrueVal=self.inputSequence[len(self.inputSequence)-1]
    AbsolutePercentageError = np.abs(trueVal- predVal) / (2*np.abs(trueVal-pastTrueVal)+0.00001)
    if saturation:
      AbsolutePercentageError = min(1,AbsolutePercentageError)



    return AbsolutePercentageError

  def handleRecord(self, inputData):


    # Get the value
    value = inputData["value"] # value: float
    self.updatePastData(value) # re-calcucate mean, var for past 1000 samples

    rawScore=0
    '''
    Step1: Anomaly detection using previous prediction at (t_now - 1) and current input value at (t_now)
    '''


    '''
    Step2: Network training
    '''

    nValue = self.normalize(value)
    # nInputSequenceBatch_forBPTT: FloatTensor(seq_len,batch_size,input_size)
    nInputSeqBatch_forBPTT = self.getInputSequenceBatchAsTensor()

    #self.train()
    self.updateInputSequence(nValue)
    #self.predict()

    finalScore = 0.0
    return (0, 1)

