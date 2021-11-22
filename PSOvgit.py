"""
PSO :

Sylvain JANNIN
Evrard EMONOT--DE CAROLIS
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class PSO:

    def __init__(self, nb_particle , new_weights, val_inertia, spacing, param_hidden, activation_function, initX, initY):  
        
        self.particle = nb_particle                                         #number of particles      
        self.list_weights = new_weights                                     #list of weight depending of the number of inputs
        self.dim = self.count_dimensions(new_weights)                       #number of inputs
        
        self.weights = self.generate_empty_ANN_list(nb_particle,spacing ,param_hidden ) #self.weights is a list of list of weight for each particles
        
        self.step = self.generate_empty_ANN_list(nb_particle,spacing ,param_hidden )       # is an ANN corresponding of each particles but is used to store the step weight we will apply on our list of particles
        self.best_alone = self.generate_empty_ANN_list(nb_particle,spacing ,param_hidden ) # is a copy of the list of particles but only storing the best weight of the particles
        self.best_all = self.generate_empty_ANN(nb_particle,spacing ,param_hidden )        # generating one empty ANN to then fill it with the best of all
        
        self.inertia = val_inertia                                                         #intertia value 
        
        self.activation_function = activation_function

        self.X = initX
        self.Y = initY
        
    
    def count_dimensions(self,weights):
        dimensions = 0
        for weight in (weights):
            dimensions += weight  
        return(dimensions)

    def generate_empty_ANN_list(self,nb_particle , spacing, param_hidden ):
        
        list_PSO = [] #ANN empty list
        
        
        for i in range(nb_particle): #pour chaque particle
            
            list_layer = [] # we initialise a list of layers
            list_node = [] #we initialise a list of nodes
                
            for h in range(param_hidden[0]): #for each node
                list_weight = [] #we initialise a list of weights
                for p in range(self.dim):
                    list_weight.append(random.uniform(-spacing,spacing)) #we declare one weight randomly  
                
                    
                    
                list_node.append(np.array(list_weight, dtype=float)) #we add the list of weights to a node
            list_layer.append(np.array(list_node, dtype=float)) #we add a list of node to the layer 
            
            
            for k in range(1,len(param_hidden)): #for each layer
                
                list_node = [] #we initialise a list of nodes
                
                for l in range(param_hidden[k]): #for each node
                    list_weight = [] #we initialise a list of weights
                    
                    for j in range(len(list_layer[k-1])): #for each nde on the previous layer
                        list_weight.append(random.uniform(-spacing,spacing)) #we declare one weight randomly  
                        
                    list_node.append(np.array(list_weight, dtype=float)) #we add the list of weights to a node
                list_layer.append(np.array(list_node, dtype=float)) #we add a list of node to the layer 
                
            list_PSO.append(list_layer) #we add a list de layer
                    
    
        return(np.array(list_PSO)) #we add the ANN

    def generate_empty_ANN(self,nb_particle, spacing, param_hidden ):
        
        list_layer = [] # we initialise a list of layers
        list_node = [] #we initialise a list of nodes
            
        for h in range(param_hidden[0]): #for each node
            list_weight = [] #we initialise a list of weights
            for p in range(self.dim):
                list_weight.append(random.uniform(-spacing,spacing)) #we declare one weight randomly  
            
                
                
            list_node.append(np.array(list_weight, dtype=float)) #we add the list of weights to a node
        list_layer.append(np.array(list_node, dtype=float)) #we add a list of node to the layer 
        
        
        for k in range(1,len(param_hidden)): #for each layer
            
            list_node = [] #we initialise a list of nodes
            
            for l in range(param_hidden[k]): #for each node
                list_weight = [] #we initialise a list of weights
                
                for j in range(len(list_layer[k-1])): #for each nde on the previous layer
                    list_weight.append(random.uniform(-spacing,spacing)) #we declare one weight randomly  
                    
                list_node.append(np.array(list_weight, dtype=float)) #we add the list of weights to a node
            list_layer.append(np.array(list_node, dtype=float)) #we add a list of node to the layer 
            
            
        return(np.array(list_layer))
    
    def display_ANN(self):
        for i in range(len(self.weights)):
            print("particle n°", i,end='\n\n\n')
            
            for j in range(len(self.weights[i])):
                print("layer n°",j,end='\n\n')
                
                for k in range(len(self.weights[i][j])):
                    print("node n°",k,end='\n\n')
                    
                    for l in range(len(self.weights[i][j][k])):
                        print(self.weights[i][j][k][l])
                    print("\n")
              

    def update_weights(self): #update the weights for each particle  
        self.update_step()    #update the step to correctly update the weights
        for i in range(self.particle):
            self.weights[i] = np.add(self.weights[i], self.step[i])   #update the weight wuth the PSO logic : take the actual position and add the new stape calculated in update_step()    
        
    
    def update_step(self):  #print to check things
        for i in range(self.particle):
            c = 1.193                  # 1/2 + ln(2), best value found in paper
            
            #since we use np.array we can't write the formula in one line which is : step = inertia_component + cognitive_component + social_component
 
            inertia_component = np.multiply(self.inertia, self.step[i])             # inertia_component = self.inertia * self.step[i]

            cognitive_component = np.subtract(self.best_alone[i], self.weights[i])  #update the step regarding its personal best 
            cognitive_component = np.multiply(c, cognitive_component)               # cognitive_component = c*(self.best_alone[i] - self.weights[i])

            social_component = np.subtract(self.best_all, self.weights[i])          #update the step regarding the global best
            social_component = np.multiply(c, social_component)                     #social_component = c*(self.best_all - self.weights[i])

            temp = np.add(inertia_component, cognitive_component)    #store temporary the value in temp 
            self.step[i] = np.add(temp, social_component)


    def update_best(self):
        loss_best_all = self.forward_loss(self.best_all)                         #compute forward of this ANN, return the output of the loss function for the best ALL
        for particle in range(self.particle):                                    #do a for loop for each particles
            loss_best_alone = self.forward_loss(self.best_alone[particle])          #Compute loss function of the best alone/personal
            loss_actual = self.forward_loss( self.weights[particle])                #compute loss function of the actual weight
            if(loss_actual < loss_best_alone):                                      #If the current weights are better than the personal best
                self.best_alone[particle] = self.weights[particle]                      #Change the best alone for the actual weights
                loss_best_alone = loss_actual
                if(loss_actual < loss_best_all):                                        #if the loss function of the actual weights are better than the global best weights 
                    self.best_all = self.weights[particle]                                  #Change the best all for the actual weights
                    loss_best_all = loss_actual

    # activation function
    def sigmoid(self,feature, weight):  
        z = np.dot(feature, weight)     
        return (1 / (1 + np.exp(-z)))   

    def tanh(self, feature, weight):
        z = np.dot(feature, weight)
        return(np.tanh(z))

    def reLu(self, feature, weight):
        z = np.dot(feature, weight)
        z = z * (z > 0)
        return z
    
    def forward(self, ANN_particle):     #forward propagation for one particle
        Inputs = [self.X]                 #list with all the inputs store
        for layer in range(len(ANN_particle)):   #for 1 layer in all the ANN
            Input = np.array(Inputs[-1])          #Input is the last cell of Inputs          
            if(self.activation_function == "tanh"):
                out = self.tanh(Input,ANN_particle[layer].T)    #call the activation function - tanh
                out = (out +1)/2                                #with tanh we have to add an offset because tanh has value from -1 to 1 and loss function works from 0 to 1
                out = np.array(out)                             #set as np.array

            elif(self.activation_function == "ReLU"):
                out = self.reLu(Input,ANN_particle[layer].T)    #call the activation function - reLu
                out = np.array(out)                             #set as np.array
            else:
                out = self.sigmoid(Input,ANN_particle[layer].T)  #call the activation function - sigmoid
                out = np.array(out)                     #set as np.array
            Inputs.append(out)                      #Inputs take the output as input for the next layer        
        return(Inputs[-1])                          #return the output activation function for one particle 

    def loss(h, y): #loss function        
        h = h.T[0]  
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def forward_loss(self, ANN_particle):                             #forward propagation for one particle
        output = self.forward(ANN_particle)                           #call forward propagation
        loss_output = PSO.loss(output, np.array(self.Y))    #call the loss function for the output of the ANN
        return(loss_output)                         #return the output of the loss function for one particle

    def check_results(self, best_weight):    #this code comes from lab1
    
        result = self.forward(best_weight)      #do on forward with the Weights we calculated
        f = pd.DataFrame(np.around(result, decimals=5)).join(self.Y)
        
        f['pred'] = f[0].apply(lambda x : 0 if x < 0.5 else 1)
        print("Accuracy (Loss minimization):")
        print(f.loc[f['pred']==f['Outputs']].shape[0] / f.shape[0] * 100)

        #For Confusion Matrix
        YActual = f['Outputs'].tolist()
        YPredicted =  f['pred'].tolist()

        #print("YActual", YActual)
        #print("YPredicted", YPredicted)

        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for l1,l2 in zip(YActual, YPredicted):
            if (l1 == 1 and  l2 == 1):
                TP = TP + 1
            elif (l1 == 0 and l2 == 0):
                TN = TN + 1
            elif (l1 == 1 and l2 == 0):
                FN = FN + 1
            elif (l1 == 0 and l2 == 1):
                FP = FP + 1

        print("Confusion Matrix: ")

        print("True Positives =", TP)
        print("True Negatives =", TN)
        print("False Positives=", FP)
        print("False Negatives=", FN)

        # Precision = TruePositives / (TruePositives + FalsePositives)
        # Recall = TruePositives / (TruePositives + FalseNegatives)

        if((TP + FP == 0) or (TP + FN == 0)):
            print("Traning failled")
        
        else:

            P = TP/(TP + FP)
            R = TP/(TP + FN)

            print("Precision = ", P)
            print("Recall    = ", R)

            #F-Measure = (2 * Precision * Recall) / (Precision + Recall), sometimes called F1

            F1 = (2* P * R)/(P + R)

            print("F score = ", F1)



#function use to init the data bank dataset
def init_data(file_name): 
    
    data = pd.read_csv(file_name) 
    Names = ["Input 1", "Input 2", "Input 3", "Input 4", "Outputs" ]
    data.columns = Names
    
    
    df = data.copy()
    
    X = df[["Input 1", "Input 2", "Input 3", "Input 4"]].copy()
    Y_origin = df["Outputs"].copy() 
    
    intercept = np.ones((X.shape[0], 0)) 
    print(intercept)
    X = np.concatenate((intercept, X), axis=1)
    
    return (X,Y_origin)
    
#function use to plot 4 weights of the ann (used for simple example)
def plot_weights(nodes,boundarie):
    
    plt.xlim(-boundarie, boundarie)
    plt.ylim(-boundarie, boundarie)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()   
    
    x_plot_1 = []
    y_plot_1 = []
    
    x_plot_2 = []
    y_plot_2 = []
    
    for i in range(len(nodes)):
        
        x_plot_1.append(nodes[i][0])
        y_plot_1.append(nodes[i][1])
        x_plot_2.append(nodes[i][2])
        y_plot_2.append(nodes[i][3])
        
        
    plt.scatter(x_plot_1, y_plot_1)
    plt.scatter(x_plot_2, y_plot_2)
    plt.show()

def count_dimension(array):#count the number of dimensions in array
    sum = 0
    for number in array:
        sum += number
    return(sum)
    


if __name__ == "__main__":

    X, Y = init_data("data_banknote_authentication.txt") #X is the feature and Y is the classification

    #parameters
    inertia = 0.721             #the best intertia values from paper -> 1/(2*ln(2)) = 0.721
    spacing = 1                 #Weights spacing 
    Layers = [4]                #setup the number of perceptron for each layer. The first cel is the inputlayer, the other are hiddens layers 
    Layers.append(1)            #add the output layer
    epochs = 100                #number of epoch for the tranning
    
    nb_dimension = count_dimension(Layers)
    #nb_particle = 40
    nb_particle = 10 + 2*int((nb_dimension)**(1/2))      #number of particle for the PSO -> 10 + 2*sqrt(number of dimension) #or 40
    activation_function = "sigmoid"                      #"ReLU", "tanh" or "sigmoid"
    
    my_pso = PSO(nb_particle, [len(X[0])], inertia, 1, Layers, activation_function, X, Y) #creating the PSO
    
  
    #Traning
    print("Start traning")
    for epoch in range(epochs):                
        my_pso.update_best()        #update the weight by doing a forward propagation and calculating the loss function
        my_pso.update_weights()     #update the weight with the PSO's logic
    print("Tranning is over")

    #take the best weights it found during the traning and do a last forward propagation to check the accuracy and print the confusion matrix
    my_pso.check_results(my_pso.best_all)
    print("the best ANN found is :\n", my_pso.best_all)
