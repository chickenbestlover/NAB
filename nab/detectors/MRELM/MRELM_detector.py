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
from FOS_ELM import FOSELM
from nab.detectors.base import AnomalyDetector
from nupic.algorithms import anomaly_likelihood
import math
"""
Implementation of the online-sequential extreme learning machine

Reference:n
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

def sigmoidActFunc(features, weights, bias):
  assert(features.shape[1] == weights.shape[1])
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = weights.shape
  V = np.dot(features, np.transpose(weights))
  for i in range(numHiddenNeuron):
    V[:, i] += bias[0, i]
  H = 1 / (1+np.exp(-V))
  return H


def linear_recurrent(features, inputW,hiddenW,hiddenA, bias, markovInput):
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = inputW.shape
  recurrentInput = np.dot(hiddenA,hiddenW)
  V = np.dot(features, np.transpose(inputW)) + recurrentInput + markovInput
  for i in range(numHiddenNeuron):
    V[:, i] += bias[0, i]

  return V, recurrentInput

def sigmoidAct_forRecurrent(features,inputW,hiddenW,hiddenA,bias):
  (numSamples, numInputs) = features.shape
  (numHiddenNeuron, numInputs) = inputW.shape
  V = np.dot(features, np.transpose(inputW)) + np.dot(hiddenA,hiddenW)
  for i in range(numHiddenNeuron):
    V[:, i] += bias[0, i]
  H = 1 / (1 + np.exp(-V))
  return H

def sigmoidActFunc(V):
  H = 1 / (1+np.exp(-V))
  return H


class MRELMDetector(AnomalyDetector):
  #def __init__(self, inputs, outputs, numHiddenNeurons, activationFunction,BN=True,AE=True,ORTH=True,
  #             inputWeightForgettingFactor=0.999,
  #            outputWeightForgettingFactor=0.999,
  #             hiddenWeightForgettingFactor=0.999):
  def __init__(self, *args, **kwargs):
    super(MRELMDetector, self).__init__(*args, **kwargs)

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
      self.anomalyLikelihood = anomaly_likelihood.AnomalyLikelihood(
        learningPeriod=numentaLearningPeriod,
        estimationSamples=self.probationaryPeriod - numentaLearningPeriod,
        reestimationPeriod=100
      )

    self.activationFunction = "sig"
    self.inputs = 100
    self.outputs = 1
    self.numHiddenNeurons = 25

    # input to hidden weights
    self.inputWeights  = np.random.random((self.numHiddenNeurons, self.inputs))
    # hidden layer to hidden layer wieghts
    self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
    # initial hidden layer activation
    self.initial_H = np.random.random((1, self.numHiddenNeurons)) * 2 -1
    self.H = self.initial_H
    self.BN = True
    self.AE = True
    self.ORTH = True
    # bias of hidden units
    self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1
    # hidden to output layer connection
    self.beta = np.random.random((self.numHiddenNeurons, self.outputs))

    # auxiliary matrix used for sequential learning
    self.M = inv(0.00001 * np.eye(self.numHiddenNeurons))

    self.inputSequence = [0.0] * self.inputs
    self.forgettingFactor = 0.915

    if self.AE:
      self.inputAE = FOSELM(inputs = self.inputs,
                            outputs = self.inputs,
                            numHiddenNeurons = self.numHiddenNeurons,
                            activationFunction = self.activationFunction,
                            BN = True,
                            forgettingFactor=0.9995,
                            ORTH = self.ORTH
                            )

      self.hiddenAE = FOSELM(inputs = self.numHiddenNeurons,
                             outputs = self.numHiddenNeurons,
                             numHiddenNeurons = self.numHiddenNeurons,
                             activationFunction=self.activationFunction,
                             BN = True,
                             forgettingFactor=0.9995,
                             ORTH = self.ORTH
                             )

    self.initializePhase(lamb=0.00001)
    
    self.sigma = 0.1
    self.minForget = 0.85
    self.recurrentInput = np.dot(self.initial_H, self.hiddenWeights)
    self.RISequence=[self.recurrentInput] * self.inputs


  def batchNormalization(self, H, scaleFactor=1, biasFactor=0):

    H_normalized = (H-H.mean())/(np.sqrt(H.var() + 0.000001))
    H_normalized = scaleFactor*H_normalized+biasFactor

    return H_normalized

  def __calculateInputWeightsUsingAE(self, features):
    self.inputAE.train(features=features,targets=features)
    return self.inputAE.beta

  def __calculateHiddenWeightsUsingAE(self, features):
    self.hiddenAE.train(features=features,targets=features)
    return self.hiddenAE.beta

  def calculateHiddenLayerActivation(self, features):
    """
    Calculate activation level of the hidden layer
    :param features feature matrix with dimension (numSamples, numInputs)
    :return: activation level (numSamples, numHiddenNeurons)
    """

    if self.AE:
      self.inputWeights = self.__calculateInputWeightsUsingAE(features)

      self.hiddenWeights = self.__calculateHiddenWeightsUsingAE(self.H)

    V,recurrentInput = linear_recurrent(features=features,
                         inputW=self.inputWeights,
                         hiddenW=self.hiddenWeights,
                         hiddenA=self.H,
                         bias= self.bias,
                         markovInput=self.RISequence[0])
    self.recurrentInput=recurrentInput
    self.RISequence.append(self.recurrentInput)
    self.RISequence.pop(0)
    if self.BN:
      V = self.batchNormalization(V)
    self.H = sigmoidActFunc(V)

    return self.H


  def initializePhase(self, lamb=0.0001):
    """
    Step 1: Initialization phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """



    #self.bias = np.random.random((1, self.numHiddenNeurons)) * 2 - 1

    self.bias = np.zeros((1, self.numHiddenNeurons))

    self.M = inv(lamb*np.eye(self.numHiddenNeurons))
    self.beta = np.zeros([self.numHiddenNeurons,self.outputs])

    # randomly initialize the input->hidden connections
    self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
    self.inputWeights = self.inputWeights * 2 - 1

    if self.AE:
     self.inputAE.initializePhase(lamb=0.00001)
     self.hiddenAE.initializePhase(lamb=0.00001)
    else:
      # randomly initialize the input->hidden connections
      self.inputWeights = np.random.random((self.numHiddenNeurons, self.inputs))
      self.inputWeights = self.inputWeights * 2 - 1

      if self.ORTH:
        if self.numHiddenNeurons > self.inputs:
          self.inputWeights = orthogonalization(self.inputWeights)
        else:
          self.inputWeights = orthogonalization(self.inputWeights.transpose())
          self.inputWeights = self.inputWeights.transpose()

      # hidden layer to hidden layer wieghts
      self.hiddenWeights = np.random.random((self.numHiddenNeurons, self.numHiddenNeurons))
      self.hiddenWeights = self.hiddenWeights * 2 - 1
      if self.ORTH:
        self.hiddenWeights = orthogonalization(self.hiddenWeights)


  def reset(self):
    self.H = self.initial_H

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

  def updateRISequence(self):
    self.RISequence.append(self.recurrentInput)
    self.RISequence.pop(0)

  def normalize(self, input):

    return (input-self.mean)/(self.var+0.00001)

  def reconstruct(self, input):

    return input*self.var + self.mean

  def getInputSequenceAsArray(self):

    #print self.inputSequence
    inputSeqArr1D = np.asarray(self.inputSequence)
    inputSeqArr2D = np.reshape(inputSeqArr1D,(1,len(inputSeqArr1D)))
    return inputSeqArr2D

  def train(self, features, targets):
    """
    Step 2: Sequential learning phase
    :param features feature matrix with dimension (numSamples, numInputs)
    :param targets target matrix with dimension (numSamples, numOutputs)
    """
    (numSamples, numOutputs) = targets.shape
    assert features.shape[0] == targets.shape[0]

    H = self.calculateHiddenLayerActivation(features)
    Ht = np.transpose(H)

    self.RLS_k = np.dot(self.M, Ht)/(self.forgettingFactor + np.dot(H, np.dot(self.M, Ht)))
    self.RLS_e = targets - np.dot(H, self.beta)
    self.beta = self.beta + np.dot(self.RLS_k, self.RLS_e)
    self.forgettingFactor = 1 - (1 - np.dot(H,self.RLS_k))*pow(self.RLS_e,2)/self.sigma
    self.forgettingFactor= max(self.minForget,self.forgettingFactor)
    #print "f=", self.forgettingFactor
    self.M = 1 / (self.forgettingFactor) * (self.M - np.dot(self.RLS_k, np.dot(H, self.M)))

#    try:
#      scale = 1/(self.forgettingFactor)
#      self.M = scale*self.M - np.dot(scale*self.M,
#                       np.dot(Ht, np.dot(
#          pinv(np.eye(numSamples) + np.dot(H, np.dot(scale*self.M, Ht))),
#          np.dot(H, scale*self.M))))
#
#      self.beta = (self.forgettingFactor)*self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, (self.forgettingFactor)*self.beta)))
      #self.beta = self.beta + np.dot(self.M, np.dot(Ht, targets - np.dot(H, self.beta)))


#    except np.linalg.linalg.LinAlgError:
#      print "SVD not converge, ignore the current training cycle"
    # else:
    #   raise RuntimeError

  def predict(self, features):
    """
    Make prediction with feature matrix
    :param features: feature matrix with dimension (numSamples, numInputs)
    :return: predictions with dimension (numSamples, numOutputs)
    """
    H = self.calculateHiddenLayerActivation(features)
    prediction = np.dot(H, self.beta)
    return prediction

  def computeRawAnomaly(self, trueVal, predVal,saturation=True):
    AbsolutePercentageError = np.abs(trueVal- predVal) / (np.abs(trueVal)+0.00001)
    if saturation:
      AbsolutePercentageError = max(1,AbsolutePercentageError)
    return AbsolutePercentageError

  def handleRecord(self, inputData):

    finalScore = 0.0
    # Get the value

    value = inputData["value"]
    self.updatePastData(value)

    nValue = self.normalize(value)

    inputFeatures = self.getInputSequenceAsArray()
 #   print inputFeatures
    nPredValue = self.predict(inputFeatures)
#    print np.array([[value]])

    self.train(features=inputFeatures,targets=np.array([[nValue]]))
    self.updateInputSequence(nValue)
    predValue = self.reconstruct(nPredValue)
    rawScore = self.computeRawAnomaly(trueVal=value,predVal=predValue, saturation=True)
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

  def handleRecordForEnsenble(self, inputData):

    finalScore = 0.0
    # Get the value

    value = inputData["value"]
    self.updatePastData(value)

    nValue = self.normalize(value)

    inputFeatures = self.getInputSequenceAsArray()
    #   print inputFeatures
    nPredValue = self.predict(inputFeatures)
    #    print np.array([[value]])

    self.train(features=inputFeatures, targets=np.array([[nValue]]))
    self.updateInputSequence(nValue)
    predValue = self.reconstruct(nPredValue)


    return predValue
