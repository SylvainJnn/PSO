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


    def __init__(self, nb_particule , new_weights,val_inertia,spacing,param_hidden,initX,initY):  
        
        self.particule = nb_particule   #number of particles      
        self.list_weights = new_weights  #list of weight depending of the number of inputs
        self.dim = self.count_dimensions(new_weights) #number of inputs
        
        
        self.values = self.generate_empty_ANN_list(nb_particule,spacing ,param_hidden ) #values is the list of weight for each particles
        
        self.step = self.generate_empty_ANN_list(nb_particule,spacing ,param_hidden ) # is ANN corresponding of each particules but is used to store the step weight we will apply on our list of particles
        self.best_alone = self.generate_empty_ANN_list(nb_particule,spacing ,param_hidden ) # is a copy of the list of particules but only storing the best value of the particules
        self.best_all = self.generate_empty_ANN(nb_particule,spacing ,param_hidden ) # generating one empty ANN to then fill it with the best of all
        
        self.inertia = val_inertia
        
        self.X = initX
        self.Y = initY
        
    
    def count_dimensions(self,weights):
        dimensions = 0
        for weight in (weights):
            dimensions += weight  
        return(dimensions)


    def generate_empty_ANN_list(self,nb_particule , spacing, param_hidden ):
        
        list_PSO = [] #ANN empty list
        
        
        for i in range(nb_particule): #pour chaque particule
            
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

    
    
    def generate_empty_ANN(self,nb_particule, spacing, param_hidden ):
        
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
        for i in range(len(self.values)):
            print("particule n°", i,end='\n\n\n')
            
            for j in range(len(self.values[i])):
                print("layer n°",j,end='\n\n')
                
                for k in range(len(self.values[i][j])):
                    print("node n°",k,end='\n\n')
                    
                    for l in range(len(self.values[i][j][k])):
                        print(self.values[i][j][k][l])
                    print("\n")
              

    def update_values(self):
        
        self.update_step()
        for i in range(self.particule):
            self.values[i] = np.add(self.values[i], self.step[i])            
        
    
    def update_step(self):  #print to check things
        for i in range(self.particule):
            best_c = 1.193
            c=best_c
            
            inertia_component = np.multiply(self.inertia, self.step[i])

            cognitive_component = np.subtract(self.best_alone[i], self.values[i])
            cognitive_component = np.multiply(c, cognitive_component)

            social_component = np.subtract(self.best_all, self.values[i])
            social_component = np.multiply(c, social_component)

            tpn = np.add(inertia_component, cognitive_component)
            self.step[i] = np.add(tpn, social_component)


    def update_best(self):
        loss_best_all = self.forward(self.best_all)     #compute forward of this ANN, return the Output of the lossfunction for the best ALL
        for particule in range(self.particule):
            loss_best_alone = self.forward(self.best_alone[particule])    #Compute loss function of the best alone/personal
            loss_actual = self.forward( self.values[particule])            #compute loss function of the actual weight
            if(loss_actual < loss_best_alone):                                 #If the /current weights are better than the personal best
                self.best_alone[particule] = self.values[particule]            #Change the best alone of the actual weights
                loss_best_alone = loss_actual
                if(loss_actual < loss_best_all):                               #if the loss function of the actual weights are better than the global best weights 
                    self.best_all = self.values[particule]                     #change
                    loss_best_all = loss_actual

    #to update it has to do a forward propagation then instead of looking for minimum, look for maximum (accuracy)
    #function ... --> forawrd --> check aqsccuracy --> % --> look for maximum
    

    def sigmoid(self,feature, weight):
        z = np.dot(feature, weight)     # préactivation
        return (1 / (1 + np.exp(-z)))   # activation 


    def check_accuracy(self, X, Y, weight):
        output = self.sigmoid(X, weight)
        output = np.round(output)
        check_true = 0
        for i in range(X.shape[0]):
            if(output[i] == Y[i]):
                check_true += 1
        return((check_true)/Y.shape[0])
    
  
    def forward(self, tab):
        Inputs = [self.X]
        for layer in range(len(tab)):
            Input = np.array(Inputs[-1])
            
            out = self.sigmoid(Input,tab[layer].T)
            out = np.array(out)
            Inputs.append(out)
        
        loss_output = PSO.loss(Inputs[-1], np.array(self.Y))
        
        return(loss_output)
    
    
    def forward_check(self, tab):
        Inputs = [self.X]
        for layer in range(len(tab)):
            Input = np.array(Inputs[-1])
            out = self.sigmoid(Input,tab[layer].T)
            out = np.array(out)
            Inputs.append(out)
        
        return(Inputs[-1])
    

    def loss(h, y):
        
        h = h.T[0]
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


    def check_results(self, best_weight):    #this code comes from the first lab1
    
        result = self.forward_check( best_weight)      #do on forward with the Weights we calculated
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

        print("True Positives= ", TP)
        print("True Negatives= ", TN)
        print("False Positives=", FP)
        print("False Negatives=", FN)

        # Precision = TruePositives / (TruePositives + FalsePositives)
        # Recall = TruePositives / (TruePositives + FalseNegatives)


        P = TP/(TP + FP)
        R = TP/(TP + FN)

        print("Precision = ", P)
        print("Recall = ", R)

        #F-Measure = (2 * Precision * Recall) / (Precision + Recall), sometimes called F1

        F1 = (2* P * R)/(P + R)

        print("F score = ", F1)

#fucntions from the code from lab1 to open the data and use it
def init_data():   
        
        data = pd.read_csv("iris.csv") #init les données
        print("Dataset size")
        print("Rows {} Columns {}".format(data.shape[0], data.shape[1]))
        print("Columns and data types")
        pd.DataFrame(data.dtypes).rename(columns = {0:'dtype'})#pour mieux s'en servir ? *
        
        df = data.copy() #pourquoi
        
        df["resultat"] = df["species"].apply(lambda x : 1 if(x == "setosa") else 0)
        #X = df[["sepal_length", "petal_length", "petal_width"]].copy()
        X = df[["sepal_length", "petal_length"]].copy()
        Y = df["resultat"].copy() # should give another name
        return(X,Y)


#function use to init the data bank dataset
def init_data2(file_name): 
    
    data = pd.read_csv(file_name) 
    Names = ["Input 1", "Input 2", "Input 3", "Input 4", "Outputs" ]
    data.columns = Names
    
    
    df = data.copy()
    
    X = df[["Input 1", "Input 2", "Input 3", "Input 4"]].copy()
    Y_origin = df["Outputs"].copy() 
    
    intercept = np.ones((X.shape[0], 0)) 
    print(intercept)
    X = np.concatenate((intercept, X), axis=1)
    
    print("init done")
    
    
    return (X,Y_origin)

#function use to plot 2 variables of the dataset (used for simple example)
def plot_dataset(X,Y):
        
    x_plot_1 = []
    y_plot_1 = []
    
    x_plot_2 = []
    y_plot_2 = []
    
    for i in range(len(X)):
        if Y[i] == 0:
            x_plot_1.append(X[i][0])
            y_plot_1.append(X[i][1])
        else: 
            x_plot_2.append(X[i][0])
            y_plot_2.append(X[i][1])
        
    plt.scatter(x_plot_1, y_plot_1)
    plt.scatter(x_plot_2, y_plot_2)
    plt.show()
   
  
#same funciton as above, but use to plot the convergion of a PSO with one perceptron for 4 inputs
def plot_perceptrons2(nodes,boundarie):
    
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


#uses to put bondaries to the grphic and plot the weights that converging 
def plot_perceptrons(nodes,boundarie):
    
    plt.xlim(-boundarie, boundarie)
    plt.ylim(-boundarie, boundarie)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()   
    
    x_plot_1 = []
    y_plot_1 = []
    
    for i in range(len(nodes)):
        
        x_plot_1.append(nodes[i][0])
        y_plot_1.append(nodes[i][1])
        
        
    plt.scatter(x_plot_1, y_plot_1)
    plt.show()

    


if __name__ == "__main__":

    X, Y = init_data2("data_banknote_authentication.txt") #•REPLACE X to x if you want to use iris dataset

    best_inertia = 0.721
    
    #UNCOMMENT IF YOU WANT TO USE IRIS DATASET
    #X = x.to_numpy()
    
    
    print("X", X[0])
    perceptron = PSO(60, [len(X[0])], best_inertia,1,[2,3,1],X,Y) #creatingg the PSO
    
    print("---WEIGHT FOR ANN VALUES----")
    print(perceptron.values)  
    
    print('type tableau de perceptron', type(perceptron.values))
    print('type tableau de layer', type(perceptron.values[0]))
    print('type tableau de node', type(perceptron.values[0][0]))
    print('type tableau de weifht', type(perceptron.values[0][0][0]))
    print()
    print('type weight', type(perceptron.values[0][0][0][0]))
    print('type weight', perceptron.values[0][0][0])
    print('type weight', perceptron.values[0][0])
    
    """
    perceptron.display_ANN()
    plot_dataset(X, Y)
    """
    
    nb_iteration = 200
    
    #perceptron.forward(perceptron.values[1])
    
    for i in range(nb_iteration): #tranning
        
        perceptron.update_values()
        perceptron.update_best() 
        print(i/nb_iteration * 100)

    moy_loss = 0
    
    print(perceptron.best_all)
    for i in range(perceptron.particule):#check
        moy_loss = perceptron.forward(perceptron.values[i])
        print(moy_loss)
        
        
    yomec = [np.array([[-0.25897943,  0.01110068, -0.47977551, -0.7991208 ], [-0.06125956,  0.55197473, -1.15921413, -0.40161609]]), 
             np.array([[-0.55864311, -0.30252977],[-0.37617277, -0.29008031], [-0.41626232, -0.14662017]]),
             np.array([[-0.02143717, -0.50451028,  0.20899994]])]
    
    
    print(perceptron.values[1])
    print()
    print(perceptron.step[1])
    print()
    print(perceptron.best_alone[1])
    print("au dessus c'est le best alone")
    print(perceptron.best_all)
    print()
    
    perceptron.check_results(perceptron.best_all)
    #perceptron.forward2(perceptron.best_all)
    
    """
    
    [array([[-0.25897943,  0.01110068, -0.47977551, -0.7991208 ],
        [-0.06125956,  0.55197473, -1.15921413, -0.40161609]])
 array([[-0.55864311, -0.30252977],
        [-0.37617277, -0.29008031],
        [-0.41626232, -0.14662017]])
 array([[-0.02143717, -0.50451028,  0.20899994]])]
    """
     
    """   
    
    for i in range(perceptron.particule):
        #print("je suis à la case :", perceptron.values[i])
        check = perceptron.forward(perceptron.values[i])
        #print("je suis une froward et j'aimes les algues ")
        print(check)
    
    
        
        
            
         
     
    
    print("----WEIGHT FOR ONE PERCEPTRON VALUES-----",perceptron.values)

        
    
    

    test = perceptron.values
    
    
    for i in range(nb_iteration):
        perceptron.update_values()
        perceptron.update_best_allP(X,Y)
        #print("en cours trainning :", i/nb_iteration*100,"%")
        
        #UNCOMMENT IF YOU WANT TO PLOT FOR ONE PERCEPTRON
        #plot_perceptrons2(perceptron.values,4)

    moyenne = 0
    moy_loss = 0
    for i in range(perceptron.particule):
        #print("je suis à la case :", perceptron.values[i])
        moyenne +=  perceptron.check_accuracy(X,Y, perceptron.values[i])
        moy_loss = perceptron.cross(X,Y,perceptron.values[i])
        print(moy_loss)
        #print("en cours check :", i/perceptron.particule*100,"%")

    print("RESULTAT")
    print(moyenne/perceptron.particule)
    print(perceptron.best_all)
    
    """
    
"""

intertie = 0.8188947556125247 --> 0.7


"""
