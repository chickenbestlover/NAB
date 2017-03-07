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
from nab.detectors.FOSELM.FOSELM_detector import FOSELMDetector
import math
SPATIAL_TOLERANCE = 0.05
class EFOSELMDetector(AnomalyDetector):

  def __init__(self, *args, **kwargs):
    super(EFOSELMDetector, self).__init__(*args, **kwargs)

    self.numDetectors = 5
    self.detectors = [FOSELMDetector(*args, **kwargs)] * self.numDetectors

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
  def computeRawAnomaly(self, trueVal, predVal,saturation=True):
    AbsolutePercentageError = np.abs(trueVal- predVal) / (np.abs(trueVal)+0.00001)
    if saturation:
      AbsolutePercentageError = max(1,AbsolutePercentageError)
    return AbsolutePercentageError

  def handleRecord(self, inputData):
      predictions = []
      value = inputData["value"]
      for i in range(0,self.numDetectors):
          tempPred = self.detectors[i].handleRecordForEnsenble(inputData)
          predictions.append(tempPred)
      predValue= sum(predictions)/self.numDetectors

      rawScore = self.computeRawAnomaly(trueVal=value, predVal=predValue, saturation=True)
      # print rawScore
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
          # print finalScore
      else:
          finalScore = rawScore

      if spatialAnomaly:
          finalScore = 1.0

      return (finalScore,)