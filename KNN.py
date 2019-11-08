from numpy.linalg import norm
from collections import Counter

class KNN():
    
    def train(self, X, Y):
        self.X_train = X
        self.Y_train = Y
        self.X_Y_train_dict = {}
        for row in range(len(self.X_train)):
            self.X_Y_train_dict[row] = self.Y_train[row]
    
    
    def predict(self, X_test, k):    
        Y_pred_key = []
        for row in range(len(X_test)):
            neighbors, test_dict = self.getKnearestNeighbors(X_test[row], k)
            max_voted = self.calculate_votes(neighbors)
            y_key = [key for key in test_dict if test_dict[key] == max_voted]
            Y_pred_key.append(y_key)
        return self.getClass(Y_pred_key)
                  
    
    def getClass(self, Y_pred_key):
        Y_pred = []
        for y in Y_pred_key:
            Y_pred.append(self.X_Y_train_dict[y[0]])
        return Y_pred
    
    def calculate_votes(self, neighbors):
        max_voted = Counter(neighbors).most_common()[0][0]
        return max_voted
        
    
    def getKnearestNeighbors(self, instance, k):
        neighbors = []
        distance = []
        index = []
        
        for row in range(len(self.X_train)): 
            distance.append(self.cosine_sim(instance, self.X_train[row]))
            index.append(row)
        
        test_dict = {index: distance for index, distance in zip(index, distance)}
        distance = sorted(distance, reverse = True)
        neighbors = distance[:k]
        return neighbors, test_dict


    def cosine_sim(self, x, y):
        cosine = np.dot(x,y) / (norm(x) * norm(y))
        return cosine

data = KNN()
data.train(X_train, Y_train)
Y_pred = data.predict(X_test, 3)
print(Y_pred)


@anvil.server.callable
def algorithm():
    return Y_pred
