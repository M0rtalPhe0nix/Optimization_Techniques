import numpy as np
import matplotlib.pyplot as plt
class Newton_GD:
  def __init__(self,data,target_label,first_derivative,second_derivative,
    learning_rate = 1,epochs = 300, 
    stop_criteria = 0.001,convergance_criteria = 0.001) -> None:

        self.data = data 
        self.target_label = target_label
        self.first_derivative = first_derivative
        self.second_derivative = second_derivative
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.stop_criteria = stop_criteria
        self.convergance_criteria = convergance_criteria
        self.costs = []
  def optimize(self):
    m = self.data.shape[0]
    n = self.data.shape[1] + 1
    self.data = np.concatenate((np.ones((m,1)),self.data), axis = 1)
    grad = np.zeros((n,1))
    theta = np.zeros((n,1))
    for i in range(self.epochs):
      theta = theta - (self.learning_rate * grad.T @ np.linalg.inv(self.second_derivative(self.data))).T
      grad ,cost = self.first_derivative(self.data,self.target_label,theta)
      self.costs.append(cost)
      if(np.linalg.norm(grad) <= self.stop_criteria):return i,theta,grad
      if(i > 0):
        if (abs(self.costs[i] - self.costs[(i-1)]) < self.convergance_criteria):return i,theta,grad
    return self.epochs,grad,theta
  def loss_vs_epochs(self):
        plt.figure(figsize=(10,8))
        plt.grid()
        plt.plot(range(len(self.costs)),self.costs)
        plt.scatter(range(len(self.costs)),self.costs)

class Linear_Regression:

    def cost(target_label,predicted_label,m):
        error_vector = predicted_label.T - target_label
        cost = (np.linalg.norm(error_vector) ** 2) / (m*2)
        return cost,error_vector
    def gradient_1(data,target_label,theta):
        m = data.shape[0]
        predicted_label = data@theta
  #predicted_label = predicted_label.T
        error_vector = predicted_label - target_label
        cost = (np.linalg.norm(error_vector) ** 2) / (m*2)
        D_theta = (error_vector.T@data) / m
        return D_theta.T,cost
    def gradient_2(data):
        m = data.shape[0]
        D_theta_2 = data.T@data / m
        return D_theta_2
