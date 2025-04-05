#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This is an L-Step predicion class created based on the work of Xilin Yang's paper:
Displacement motion prediction of a landing deck for recovery operations of rotary UAVs.
Please read the paper for a more detailed explaination of the underlying theory.

The class uses a forgetting factor recursive least squares algorithm to determine the coefficients of an auto-regressive-like model
for the purpose of predicting the future state of an ocean vessel exhibiting periodic motion due to ocean waves.

"""

__author__ = "Andrew Tayler"
__version__ = "0.1"
__status__ = "Production"


import numpy as np
from scipy import empty

# L-Step predictor
class L_Step:
    def __init__(self,M,N,lam):
        '''
        Initialise L Step predictor class.

        M: int
            Number of A coefficients
        N: int
            Number of B coefficients
        lam: float
            Forgetting factor usually between 0.98-0.995
        '''
        
        self.M = M # Number of A coefficients
        self.N = N # Number of B coefficients
        self.lam = lam # Forgetting Factor
       
        # Initialise matrices/vectors that carry over between iterations
        self.P = 1e9*np.matrix(np.identity(self.M+self.N)) # Covariance matrix
        self.A = np.matrix(np.zeros([self.M+self.N,1])) # Coefficients vector
        self.e = 0
        self.sse = 0
        self.bic = 0
        self.mavg = 0
        self.mmax = 0

        self.predictions = []
        self.prediction_step = 1

        self.peaks = []
        self.peaks_step = []
       
        self.num_obs = 0
        self.obs_buffer = []
        self.obs_limit = self.M + self.N

        # Fill with buffer with zeros up to the max model order
        while(len(self.obs_buffer) < self.obs_limit):
            self.obs_buffer.insert(0,0)  

        self.update_time = 0

        print("L-Step Predictor Class Created.")

    
    def update(self,yt):
        """
        Update the model efficients of forgetting factor recursive least squares algorithm.

        y(t): float
            New data point
        """

        if len(self.obs_buffer) >= self.obs_limit:
            # Get vector of lagged input-output data in the form of
            # y(t), y(t-1), ... , y(t-m), y(t-m-1), ... , y(t-M-N+1)]
            V =  np.asmatrix(self.obs_buffer).T
            self.mavg = np.matrix.mean(V)
            self.mmax = np.matrix.max(V)
            self.dy = yt - self.obs_buffer[0]
        
            Pq = self.P
        
            # Updating matrix
            lam_inv = 1/self.lam
            K = (lam_inv*np.matmul(Pq,V))/(1+lam_inv*np.matmul(np.matmul(V.T,Pq),V))

            # Test to see how the old estimate compares to the new data
            self.e = np.asscalar(yt - np.matmul(self.A.T,V))
            self.sse += self.e**2

            # New coefficients are the old + the gain * the error
            self.A = self.A + K*self.e

            # Update inverse
            self.P = lam_inv*Pq - np.matmul(np.matmul(lam_inv*K,V.T),Pq)
        
        # Insert the new observation at the front of the line such that:
        # [y(t+1), y(t), y(t-1), ..., y(t-M-N+2)]
        self.obs_buffer.insert(0,yt)
        self.num_obs += 1

        # Drop off the oldest observation
        # [y(t), y(t-1), ..., y(t-M-N+1)]
        if len(self.obs_buffer) > self.obs_limit:
            self.obs_buffer.pop()
        

    def get_error(self):
        """
        Return the instantaneous squared error of the algorithm.
        """    

        e = self.e**2
        return e


    def calc_bic(self):

        # TODO Verify if this works properly
        sigma = self.sse/(self.num_obs-self.M-self.N)
        self.bic = np.log(sigma) + ((self.M+self.N)*np.log(self.num_obs))/self.num_obs
        print(np.log(sigma),((self.M+self.N)*np.log(self.num_obs))/self.num_obs)
        

    def predict_range(self,L):
        """
        Predict the future state of the system.

        L: int
            Number of time steps to predict 
        """
        
        if len(self.obs_buffer) >= self.obs_limit:

            y_hat_buffer = self.obs_buffer[:self.M]
            predictions = []

            for l in range(L):
                V =  np.asmatrix(y_hat_buffer+self.obs_buffer[self.M:]).T
                y_hat = np.asscalar(self.A.T*V)
                y_hat_buffer.insert(0,y_hat)
                predictions.append(y_hat)
                y_hat_buffer.pop()

            self.predictions = predictions
            self.prediction_step = self.num_obs-1
            return predictions

        else:
            return None


    def predict_range_adjusted(self,L):
        """
        Predict the future state of the system. Has added adjustment which punishes predictions
        for being far away from the observation buffer average.

        L: int
            Number of time steps to predict 
        """
        
        if len(self.obs_buffer) >= self.obs_limit:

            y_hat_buffer = self.obs_buffer[:self.M]
            predictions = []

            for l in range(L):
                V =  np.asmatrix(y_hat_buffer+self.obs_buffer[self.M:]).T
                y_hat = np.asscalar(self.A.T*V)
                y_hat_buffer.insert(0,y_hat)
                predictions.append(y_hat)
                y_hat_buffer.pop()

            # Shift predictions based on how far it is straying from the observation average. Shifting factor scales linearly into the future
            pavg = np.matrix.mean(np.asmatrix(predictions))
            avg_diff = self.mavg-pavg
            p = []
            i = float(1)
            for prediction in predictions:
                p.append(prediction + (2*avg_diff*(i/len(predictions))))
                i += 1

            self.predictions = p
            self.prediction_step = self.num_obs-1

            return p

        else:
            return None

    def find_peaks(self):
        """
        Find the peaks or local maxima in the prediction set.
        """

        if not self.predictions:
            return None

        peaks = []
        steps = []
        
        for i in range(1,len(self.predictions)-1):
            if (self.predictions[i] > self.predictions[i-1]) and (self.predictions[i] > self.predictions[i+1]):
                peaks.append(self.predictions[i])
                steps.append(i)

        if not steps:
            return None

        self.peaks = peaks
        self.peaks_step = steps

        return steps,peaks
