import numpy as np

class NNet:

    z = [] # Inputs Matrix
    a = [] # Activation Matrix, a = g(z)
    lr = 0.1 # Learning Rate

    def __init__(self, **kwargs): # Initializing inputs, thetas and biases
        self.__dict__.update(kwargs)

    # Activation Function:
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    # Derivative Sigmoid:
    def d_sigmoid(self, a):
        return a*(1-a)        # g(x) = a

    # Forward Propagation Function:
    def feedforward(self, x):
        NNet.a, NNet.z = [], []
        for b, w in zip(self.biases, self.weights):
            temp = np.dot(w, x) + b # theta1*x1 + theta2 * x2 + b1
            NNet.z.append(temp)
            x = self.sigmoid(temp)
            NNet.a.append(x)
        return x

    # Prediction Function:
    def predict(self, x):
        for b, w in zip(self.biases, self.weights):
            x = self.sigmoid(np.dot(w, x) + b)
        return x

    # Backpropagation Algorithm:
    def backprop(self, x, y):
        l = self.layer
        for i in range(-1, -l, -1):
            if i==-1:
                delta = y - self.feedforward(x)
                corr = (NNet.lr)*delta*self.d_sigmoid(NNet.a[i])
                corrections = np.dot(corr, NNet.a[i-1].T)
                self.weights[i] += corrections
                self.biases[i] += corr
            elif i==-l+1:
                delta = np.dot(self.weights[i+1].T, delta)
                corr = (NNet.lr)*delta*self.d_sigmoid(NNet.a[i])
                corrections = np.dot(corr, x.T)
                self.weights[i] += corrections
                self.biases[i] += corr
            else:
                delta = np.dot(self.weights[i+1].T, delta)
                corr = (NNet.lr)*delta*self.d_sigmoid(NNet.a[i])
                corrections = np.dot(corr, NNet.a[i-1].T)
                self.weights[i] += corrections
                self.biases[i] += corr

    # Training:
    def train(self, itr, Y, m):
        #print("Before:")
        #self.show_stat()
        for _ in range(itr):
            for i in range(m):
                self.feedforward(self.input[i].reshape(-1, 1))
                self.backprop(self.input[i].reshape(-1, 1), y[i])
        #print("After:")
        #self.show_stat()

    def show_stat(self):
        print("W:")
        print(self.weights)
        print("B:")
        print(self.biases)


# Loading Data:
data = np.genfromtxt('week_5.csv', delimiter=',')
x = data[:, 0:2]
y = data[:, 2]

# Data Scaling:
x[:,0] = x[:,0]/max(x[:,0])
x[:,1] = x[:,1]/max(x[:,1])

# Constants:
m = len(data)

# Initializing The Weights:
theta12 = np.random.rand(3, 2)
theta23 = np.random.rand(3, 3)
theta34 = np.random.rand(1, 3)
theta = np.array([theta12, theta23, theta34], dtype=object)

# Initializing The Biases:
b12 = np.random.rand(3, 1)
b23 = np.random.rand(3, 1)
b34 = np.random.rand(1, 1)
b = np.array([b12, b23, b34], dtype=object)

# Feeding Data Into The Network:
nn = NNet(input=x, weights=theta, biases=b, layer=4)

# Training The Network:
nn.train(1000, y, m)

# Making Predictions:
pt = 0
for i in range(m):
    #print("Prediction:", nn.predict(x[i].reshape(-1, 1)), "Actual:", y[i])
    check = nn.predict(x[i].reshape(-1, 1))[0][0]
    if check>=0.5 and y[i]==1:
        pt+=1
    if check<0.5 and y[i]==0:
        pt+=1

print("Accuracy:", (pt/m)*100)


x1 = float(input("Enter x1: "))
x2 = float(input("Enter x2: "))
xx = np.array([x1, x2])
result = nn.predict(xx.reshape(-1, 1))
print("Prediction:", result)

nn.show_stat()
