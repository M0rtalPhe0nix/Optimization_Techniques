import numpy as np
import matplotlib.pyplot as plt
from math import ceil
class SGD_with_Adam:

    def __init__(self,data,target_label,function,first_derivative,batch_size,
    learning_rate = 0.0005,beta1 = 0.9,beta2 = 0.1,epsilon = 1e-8,max_iterations = 300, 
    stop_criteria = 0.001,convergance_criteria = 0.001) -> None:

        self.data = data 
        self.target_label = target_label
        self.function = function
        self.first_derivative = first_derivative
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.stop_criteria = stop_criteria
        self.convergance_criteria = convergance_criteria
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.costs = []
    def Init_SGD(self,data,target_label):

        m = self.data.shape[0]
        n = self.data.shape[1] + 1
        connected_array = np.concatenate((np.ones((m,1)),self.data,self.target_label), axis = 1)
        np.random.shuffle(connected_array)
        data = connected_array[:, :-1]
        target_label = connected_array[:, [-1]]
        return data,target_label,int(m),int(n)
    
    def optimize(self):

        self.data,self.target_label,m,n = self.Init_SGD(self.data,self.target_label)
        c = ceil(m / self.batch_size)
        theta = np.zeros((n,1))
        v = np.zeros((n,1))
        mt = np.zeros((n,1))
        for i in range(self.max_iterations):
            for j in range(0,m,self.batch_size):
                predicted_label = self.data[j:j+self.batch_size]@theta
                predicted_label = predicted_label.T
                cost,error_vector = self.function(self.target_label, predicted_label,m,self.batch_size,j)
                self.costs.append(cost)
                D_theta = self.first_derivative(error_vector,self.data,m,self.batch_size,j)
                mt = self.beta1 * mt + ((1 - self.beta1) * D_theta)
                v = self.beta2 * v + (1 - self.beta2) * D_theta ** 2
                mt = mt / (1-(self.beta1**(i+1)))
                v = v / (1-(self.beta2**(i+1)))
                theta = theta - ((self.learning_rate / (np.sqrt(v) + self.epsilon)) * mt)
            if(np.linalg.norm(D_theta,2) < self.stop_criteria):break
            if(i > 0):
                if (abs(self.costs[i * c] - self.costs[(i-1) * c]) < self.convergance_criteria):break
        return theta
    
    def loss_vs_epochs(self):
        plt.figure(figsize=(10,8))
        plt.grid()
        plt.plot(range(len(self.costs)),self.costs)
        plt.scatter(range(len(self.costs)),self.costs)


class Linear_Regression:

    def cost(target_label,predicted_label,m,batch_size,batch_number):
        error_vector = predicted_label.T - target_label[batch_number:batch_number+batch_size]
        cost = (np.linalg.norm(error_vector) ** 2) / (batch_size*2)
        return cost,error_vector
    def gradient_1(error_vector,data,m,batch_size,batch_number):
        D_theta = (error_vector.T@data[batch_number:batch_number+batch_size]) / batch_size
        return D_theta.T