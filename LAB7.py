import numpy as np
from keras.layers import Conv2D, Flatten, Dense, AveragePooling2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.datasets import mnist, cifar10
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

class zadanie7_2_3:
    def exec(self):
        train, test = mnist.load_data()
        X_train, y_train = train[0], train[1]
        X_test, y_test = test[0], test[1]
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        num_layers_list = [1,2,3,4,5,6,7,8]
        results = {}
        for num_layers in num_layers_list:
            avg_accuracy = self.evaluate_model(num_layers,X_train,y_train,X_test,y_test)
            results[num_layers] = avg_accuracy
            print(f"Num layers: {num_layers}, Avg Accuracy: {avg_accuracy}")

        best_num_layers = max(results, key=results.get)
        print(f"\nBest number of layers: {best_num_layers}, Avg Accuracy: {results[best_num_layers]}")

    def build_model(num_layers, X_train, class_cnt):
        filter_cnt = 32
        learning_rate = 0.0001
        act_func = 'relu'
        kernel_size = (3,3)
        conv_rule = 'same'
        model = Sequential()
        model.add(Conv2D(input_shape=X_train.shape[1:],
                        filters=filter_cnt,
                        kernel_size=kernel_size,
                        padding=conv_rule,
                        activation=act_func))
        for _ in range(num_layers):
            model.add(MaxPooling2D((2,2))) 
            model.add(Conv2D(filters=filter_cnt,
                            kernel_size=kernel_size,
                            padding=conv_rule,
                            activation=act_func))
        model.add(MaxPooling2D((2,2))) #zadanie 7.3
        model.add(Flatten())
        model.add(Dense(class_cnt, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
        return model

    def evaluate_model(self,num_layers, X_train, y_train, X_test, y_test):
        class_cnt = np.unique(y_train).shape[0]
        accuracies = []
        for train_index, test_index in KFold(5).split(X_train):
            X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
            model = self.build_model(num_layers,X_train,class_cnt)
            model.fit(x=X_train_cv,y=y_train_cv,epochs=10,batch_size=32,verbose=2)
            accuracy = model.evaluate(X_test_cv,y_test_cv,verbose=0)
            accuracies.append(accuracy)

        return np.mean(accuracies)

class zadanie7_4:
    def build_model(num_layers, X_train, class_cnt):
        filter_cnt = 32
        learning_rate = 0.0001
        act_func = 'relu'
        conv_rule = 'same'
        kernel_size = (3,3)
        model = Sequential()
        model.add(Conv2D(input_shape=X_train.shape[1:],
                         filters=filter_cnt,
                         kernel_size=kernel_size,
                         padding=conv_rule,
                         activation=act_func))
        for _ in range(num_layers):
            model.add(MaxPooling2D((2,2)))
            model.add(Conv2D(filters=filter_cnt,
                         kernel_size=kernel_size,
                         padding=conv_rule,
                         activation=act_func))
        
        model.add(MaxPooling2D((2,2)))
        model.add(Flatten())
        model.add(Dense(class_cnt,activation='softmax'))
        model.compile(optimizer=Adam(learning_rate),loss='SparseCategoricalCrossentropy',metrics=['accuracy'])
        return model
    
    def evaluate_model(self,num_layers,X_train,y_train,X_test,y_test):
        class_cnt = np.unique(y_train).shape[0]
        accuracies = []
        for train_index, test_index in KFold(5).split(X_train):
            X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
            model = self.build_model(num_layers,X_train,class_cnt)
            model.fit(x=X_train_cv,y=y_train_cv,epochs=10,batch_size=32,verbose=2)
            accuracy = model.evaluate(X_test_cv,y_test_cv,verbose=0)
            accuracies.append(accuracy)

        return np.mean(accuracies)
    
    def get_optimal_layers_count(self,X_train,X_test,y_train,y_test):
        
        X_train = np.expand_dims(X_train,axis=-1)
        X_test = np.expand_dims(X_test,axis=-1)
        num_layers_list = [1,2,3,4]
        results = {}
        for num_layers in num_layers_list:
            avg_accuracy = self.evaluate_model(num_layers,X_train,y_train,X_test,y_test)
            results[num_layers] = avg_accuracy
        
        return max(results, key=results.get)
    
    def learn_optimal_model(self):
        (X_train,y_train), (X_test,y_test) = cifar10.load_data()
        best_layer_count = self.get_optimal_layers_count(X_train,X_test,y_train,y_test)
        model = self.build_model(best_layer_count,X_train,np.unique(y_train).shape[0])
        model.fit(x=X_train,y=y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test),verbose=2)
        loss, accuracy = model.evaluate(X_test,y_test)
        y_pred = np.argmax(model.predict(X_test),axis=-1)
        cm = confusion_matrix(y_test,y_pred)
        print(f"Test acc: {accuracy}")
        print("---------------------------------------")
        print(cm)
    
if __name__ == "__main__":
    task = zadanie7_4()
    task.learn_optimal_model()