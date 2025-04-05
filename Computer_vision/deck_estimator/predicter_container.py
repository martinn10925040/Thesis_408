#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
This class acts as a wrapper around the prediction class for the purposes of managing multiple predictors
for each degree of freedom in a pose. It also provides real time plotting functionality for the prediction
since RQT does not support plotting into the future as far as I know.
"""

__author__ = "Andrew Tayler"
__version__ = "0.1"
__status__ = "Production"

from geometry_msgs.msg import Pose
from deck_estimator.lstep import L_Step
import matplotlib.pyplot as plt
import numpy as np

class Predictor_Container:
    def __init__(self, M, N, lam):
        '''
        Initialise predictor container class.

        M: int
            Number of A coefficients
        N: int
            Number of B coefficients
        lam: float
            Forgetting factor usually between 0.98-0.995
        '''

        # Create a list to contain all 6 degres of freedom in a pose
        self.estimators = []

        self.M = M
        self.N = N
        self.lam = lam

        self.error = []

        # Plotting stuff (The comma after plots is important, dont delete it)
        self.fig, [self.ax, self.ax2] = plt.subplots(1,2)
        self.obs_plot, = self.ax.plot([], [])
        self.prediction_plot, = self.ax.plot([], [],'--')
        self.peak_plot, = self.ax.plot([], [],'o')
        self.error_plot, = self.ax2.plot([], [])

        # Initialise recursive pronys with (Model order, forgetting factor, frequency)
        for i in range(7):
            self.estimators.append(L_Step(M, N, lam))

    def update(self,pose_msg):

        # Convert pose to a list of numbers ready for the loop
        pose = []
        pose.append(pose_msg.position.x)
        pose.append(pose_msg.position.y)
        pose.append(pose_msg.position.z)
        pose.append(pose_msg.orientation.x)
        pose.append(pose_msg.orientation.y)
        pose.append(pose_msg.orientation.z)
        pose.append(pose_msg.orientation.w)

        """Disabled updating all the predictors. Only updating the Z predictor. """
        # Update all the estimators with the new pose values
        # for i in range(7):
        #     self.estimators[i].update(pose[i])

        # Just update the Z values for now
        self.estimators[2].update(pose[2])
        self.error.append(self.estimators[2].get_error())



    def predict(self,L):
        """
        Predict the future state of the system.

        L: int
            Number of time steps to predict 
        """

        # TODO predict the whole pose
        # Only predict the Z position for now
        self.estimators[2].predict_range(L)
        peaks = self.estimators[2].find_peaks()

        # Convert pose to a list of numbers ready for the loop
        pose = Pose()
        pose.position.x = 0
        pose.position.y = 0
        pose.position.z = self.estimators[2].predict_range(L)
        #print(pose.position.z)

        pose.orientation.x = 0
        pose.orientation.y = 0
        pose.orientation.z = 0
        pose.orientation.w = 0

        return pose

    def init_plot(self):
        """
        Initialise prediction plot.
        """
        self.ax.set_title("Heave Motion with Prediction")
        self.ax.set_xlabel("Samples/Number of observations")
        self.ax.set_ylabel("Vessel Z Position")
        self.ax.legend(["Observed", "Predicted"])
        self.ax.set_ylim(-5, 5)

        self.ax2.set_title('Instantaneous Error in RLS')
        self.ax2.set_ylabel("Error")
        self.ax2.set_yscale('log')
        self.ax2.set_ylim(1e-10,3)

    def update_plot(self, frame):
        """
        Update prediction plot.
        """
        # Plot observations data
        x_data = range(self.estimators[2].num_obs-len(self.estimators[2].obs_buffer),self.estimators[2].num_obs)
        y_data = list(reversed(self.estimators[2].obs_buffer))
        self.obs_plot.set_data(x_data, y_data)

        # Plot error
        if len(self.error) > 1:
            x_data = range(self.estimators[2].num_obs-len(self.error),self.estimators[2].num_obs)
            y_data = self.error
            self.error_plot.set_data(x_data, y_data)
        
        # Plot prediction
        if self.estimators[2].prediction_step is not None:
            x_data = range(self.estimators[2].prediction_step+1,self.estimators[2].prediction_step+len(self.estimators[2].predictions)+1)
            y_data = self.estimators[2].predictions
            self.prediction_plot.set_data(x_data, y_data)

        # Plot peaks
        if self.estimators[2].peaks:
            for j in range(len(self.estimators[2].peaks)):
                x_data = self.estimators[2].prediction_step + self.estimators[2].peaks_step[j]
                y_data = self.estimators[2].peaks[j]
                self.peak_plot.set_data(x_data, y_data)


        self.ax.set_xlim(self.estimators[2].num_obs-len(self.estimators[2].obs_buffer), self.estimators[2].num_obs+70)
        self.ax2.set_xlim(self.estimators[2].num_obs-len(self.estimators[2].obs_buffer), self.estimators[2].num_obs+70)

        return self.obs_plot
