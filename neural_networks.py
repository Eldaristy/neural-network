import numpy as np

class NeuralNetwork:
    def __init__(self, input_l, hidden_l, output_l, hidden_n=1, alpha=0.5):
        self.layers = []
        self.connections = []
        self.alpha = alpha #the learning rate

        self.layers.append(Layer(input_l)) #adding the input layer
        for h in range(hidden_n):
            self.layers.append(Layer(hidden_l))#adding the hidden layer(s)
            self.connections.append(Connection(
                self.layers[-2], self.layers[-1])) #adding the connection(s)
            #between the layers (ih and hh)
        self.layers.append(Layer(output_l)) #adding the output layer
        self.connections.append(Connection(
            self.layers[-2], self.layers[-1])) #adding the connection
            #between the layers (ho)
        
    def train(self, inputs, targets):
        for sample in range(len(inputs)):
            output = self.feed_forward(inputs[sample])
            gradients = self.back_propagation(output, targets[sample])
            self.gradient_descent(gradients)
    def predict(self, inputs):
        outputs = []
        for sample in range(len(inputs)):
            outputs.append(self.feed_forward(inputs[sample]))
        d = np.vectorize(self.decide)
        return d(outputs)
    def decide(self, output): #1 if >=0.5, 0 else
        print(output)
        return 0 if output < 0.5 else 1
    def feed_forward(self, inputs):
        self.layers[0].a = np.array(inputs) 
        for i in range(len(self.layers)-1): #to skip the iteration 
            #over the input layer
            self.layers[i+1].z = self.weights_sum(i+1) #Wx + B
            self.layers[i+1].a = self.sigmoid(self.layers[i+1].z) #apply the sigmoid activation function
        return self.layers[-1].a

    def back_propagation(self, output, targets):
        nabla_w = [] #derivatives of cost function with respect to all w
        nabla_b = [] #derivatives of cost function with respect to all b
        #  δC     δC     δa     δz
        # ---- = ---- * ---- * ---- (chain rule)
        #  δw     δa     δz     δw

        #same for δb (instead of δw)
        temp = np.full(self.layers[-1].count,self.cost_wrt_to_a(output, targets))
        temp = temp.reshape(1,-1)
        
        for i in range(len(self.connections)): 
            temp *= self.sigmoid_wrt_to_z(self.layers[-1-i].z)
            nabla_w.append(temp.reshape(1,-1).T*(self.weights_sum_wrt_to_w(-1-i).reshape(1,-1)))
            nabla_b.append(temp.reshape(1,-1).T)
            temp = temp.dot(self.weights_sum_wrt_to_a(-1-i))   
        return nabla_w, nabla_b
    def gradient_descent(self, nablas):
        w = nablas[0]
        b = nablas[1]
        for l in range(len(self.connections)):
            self.connections[-1-l].w -= self.alpha*w[l]
    def weights_sum(self, j):
        return self.connections[j-1].w.dot(self.layers[j-1].a) #+ self.layers[j].b
    def weights_sum_wrt_to_w(self, j): #δz_2 / δw_1
        return self.layers[j-1].a
    def weights_sum_wrt_to_a(self, j): #δz_2 / δa_1
        return self.connections[j].w
    #δz_2 / δb_1 - is just 1
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    def sigmoid_wrt_to_z(self, z): #δa_1 / δz_1
        return z*(1-z)
    
    def cost(self,o,y):
        return 0.1*np.average((o-y)**2)
    def cost_wrt_to_a(self, o,y): #δC / δa
        return o-y
class Layer:
    def __init__(self, count):
        self.count = count
        self.b = 0 #bias
        self.z = np.zeros(count) #before activation function
        self.a = np.zeros(count) #after activation function

        
class Connection: #The connection between two layers
    #which controls the weights
    def __init__(self, L_1, L_2):
        self.L_1 = L_1
        self.L_2 = L_2
        self.w = np.random.uniform(low=-1,high=1,size=(L_2.count, L_1.count))

      
NN = NeuralNetwork(2,2,1,1)
for _ in range(50):
    NN.train([[1,1],[1,1],[1,1],[0,0]],[[1],[1],[1],[1]])
preds = NN.predict([[1,1]])
print(preds)
print("done")
