# coding=utf-8
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv
from nab.detectors.base import AnomalyDetector
from nupic.algorithms import anomaly_likelihood
import math
"""
Implementation of the online-sequential extreme learning machine

Reference:
N.-Y. Liang, G.-B. Huang, P. Saratchandran, and N. Sundararajan,
“A Fast and Accurate On-line Sequential Learning Algorithm for Feedforward
Networks," IEEE Transactions on Neural Networks, vol. 17, no. 6, pp. 1411-1423
"""

# Fraction outside of the range of values seen so far that will be considered
# a spatial anomaly regardless of the anomaly likelihood calculation. This
# accounts for the human labelling bias for spatial values larger than what
# has been seen so far.
SPATIAL_TOLERANCE = 0.05


def orthogonalization(Arr):
  [Q, S, _] = np.linalg.svd(Arr)
  tol = max(Arr.shape) * np.spacing(max(S))
  r = np.sum(S > tol)
  Q = Q[:, :r]

  return Q

def linear(features, weights, bias):

  assert(features.shape[1] == weights.shape[1])
  (numSamples, numInputs) = features.shape
  #print features.shape

  (numHiddenNeuron, numInputs) = weights.shape

  V = np.dot(features, np.transpose(weights))

  for i in range(numHiddenNeuron):
    V[:, i] += bias[0, i]

  return V

def sigmoidActFunc(V):
  H = 1 / (1+np.exp(-V))
  return H



class FOSELMDetector(AnomalyDetector):
  #def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction, BN=False,forgettingFactor=0.999, ORTH = True):
  def __init__(self, *args, **kwargs):
    super(FOSELMDetector, self).__init__(*args, **kwargs)

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
      self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        learningPeriod=numentaLearningPeriod,
        estimationSamples=self.probationaryPeriod - numentaLearningPeriod,
        reestimationPeriod=100
      )

    self.activationFunction = "sig"
    self.inputs = 100
    self.outputs = 1
    self.numHiddenNeurons = 800
    # input to hidden weights
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.ORTH = True

    # bias of hidden units
    self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    # hidden to output layer connection
    self.beta = np.random.random((self.numHiddenNeurons, self.outputs))
    self.BN = True
    # auxiliary matrix used for sequential learning
    self.M = None
    self.forgettingFactor = 0.9995

    self.inputSequence = [0.0] * self.inputs

    self.initializePhase(lamb=0.00001)

  def batchNormalization(self, H, scaleFactor=1, biasFactor=0):

    H_normalized = (H - H.mean()) / (np.sqrt(H.var() + 0.0001))
    H_normalized = scaleFactor * H_normalized + biasFactor

    return H_normalized

  def calculateHiddenLayerActivation(self, features):
    """
    Calculate activation level of the hidden layer
    :param features feature matrix with dimension (numSamples, numInputs)
    :return: activation level (numSamples, numHiddenNeurons)
    """

    V = linear(features, self.inputWeights,self.bias)
    if self.BN:
      V = self.batchNormalization(V)
    H = sigmoidActFunc(V)

    return H


  def initializePhase(self, lamb=0.0001):
    """
    Step 1: Initialization phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """

    # randomly initialize the input->hidden connections
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.inputWeights = self.inputWeights * 2 - 1

    if self.ORTH:
      if self.numHiddenNeurons > self.inputs:
        self.inputWeights = orthogonalization(self.inputWeights)
      else:
        self.inputWeights = orthogonalization(self.inputWeights.transpose())
        self.inputWeights = self.inputWeights.transpose()


    self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1


    self.M = inv(lamb*np.eye(self.numHiddenNeurons))
    self.beta = np.zeros([self.numHiddenNeurons,self.outputs])

  def updateInputSequence(self, newInput):

    self.inputSequence.append(newInput)
    self.inputSequence.pop(0)

  def getInputSequenceAsArray(self):

    #print self.inputSequence
    inputSeqArr1D = np.asarray(self.inputSequence)
    inputSeqArr2D = np.reshape(inputSeqArr1D,(1,len(inputSeqArr1D)))
    return inputSeqArr2D

  def train(self, features, targets,RLS=False):
    """
    Step 2: Sequential learning phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
    (numSamples, numOutputs) = targets.shape
    assert features.shape[0] == targets.shape[0]

    H = self.calculateHiddenLayerActivation(features)
    Ht = np.transpose(H)

    if RLS:

      self.RLS_k = np.dot(np.dot(self.M,Ht),inv( self.forgettingFactor*np.eye(numSamples)+ np.dot(H,np.dot(self.M,Ht))))
      self.RLS_e = targets - np.dot(H,self.beta)
      self.beta = self.beta + np.dot(self.RLS_k,self.RLS_e)
      self.M = 1/(self.forgettingFactor)*(self.M - np.dot(self.RLS_k,np.dot(H,self.M)))
    else:

      scale = 1 / (self.forgettingFactor)
      self.M = scale * self.M - np.dot(scale * self.M,
                                       np.dot(Ht, np.dot(
                                         pinv(np.eye(numSamples) + np.dot(H, np.dot(scale * self.M, Ht))),
                                         np.dot(H, scale * self.M))))

      self.beta = (self.forgettingFactor)*self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (self.forgettingFactor)*self.beta)))
      #self.beta = (self.forgettingFactor)*self.beta + (self.forgettingFactor)*np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))
      #self.beta = self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))


  def predict(self, features):
    """
    Make prediction with feature matrix
    :param features: feature matrix with dimension (numSamples, numInputs)
    :return: predictions with dimension (numSamples, numOutputs)
    """
  #  print 'hello1'
    H = self.calculateHiddenLayerActivation(features)
   # print 'hello2'
    prediction = np.dot(H, self.beta)
    #print 'hello3'
    return prediction

  def computeRawAnomaly(self, trueVal, predVal):
    AbsolutePercentageError = np.abs(trueVal- predVal) / (np.abs(trueVal)+0.00001)
    return AbsolutePercentageError

  def handleRecord(self, inputData):

    finalScore = 0.0
    # Get the value

    value = inputData["value"]
    inputFeatures = self.getInputSequenceAsArray()
 #   print inputFeatures
    predValue = self.predict(inputFeatures)
#    print np.array([[value]])
    self.train(features=inputFeatures,targets=np.array([[value]]),RLS=True)
    self.updateInputSequence(value)

    rawScore = self.computeRawAnomaly(trueVal=value,predVal=predValue)
    #print rawScore
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

    if self.useLikelihood:
      # Compute log(anomaly likelihood)
      anomalyScore = self.anomalyLikelihood.anomalyProbability(
        inputData["value"], rawScore, inputData["timestamp"])
      logScore = self.anomalyLikelihood.computeLogLikelihood(anomalyScore)
      finalScore = logScore
      #print finalScore
    else:
      finalScore = rawScore

    if spatialAnomaly:
      finalScore = 1.0


    return (finalScore,)
