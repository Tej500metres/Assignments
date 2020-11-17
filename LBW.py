'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import numpy as np
import pandas as pd
        
class NN:
    def sigmoid(self,z):
        a=1/(1+np.exp(-z))
        cache=z
        return a,cache
    def relu(self,z):
        a=np.maximum(0,z)
        cache=z
        return a,cache
    def fit(self):	
        '''Function that trains the neural network by taking x_train and y_train samples as input'''
        
        #importing dataset
        dataset = pd.read_csv('LBW_Dataset.csv')
        #missing values
        dataset['Age']=dataset['Age'].fillna(round(dataset['Age'].mean()))
        dataset['Weight']=dataset['Weight'].fillna(dataset['Weight'].mean())
        dataset['Education']=dataset['Education'].fillna(dataset['Education'].mode()[0])
        dataset['Delivery phase']=dataset['Delivery phase'].fillna(dataset['Delivery phase'].mode()[0])
        dataset['HB']=dataset['HB'].fillna(dataset['HB'].mean())
        dataset['BP']=dataset['BP'].fillna(dataset['BP'].mode()[0])
        dataset['Residence']=dataset['Residence'].fillna(dataset['Residence'].mode()[0])
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, -1].values
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
    
        def relu_backward(da,cache):
            z=cache
            dz=np.array(da,copy=True)
            dz[z<=0]=0
            assert(dz.shape==z.shape)
            return dz
        def sigmoid_backward(da,cache):
            z=cache
            s,_=self.sigmoid(z)
            dz=da*s*(1-s)
            assert(dz.shape==z.shape)
            return dz
        def intialize_parameters_deep(layer_dims):
            parameters={}
            l=len(layer_dims)
            for i in range(1,l):
                parameters['w'+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])/np.sqrt(layer_dims[i-1])
                parameters['b'+str(i)]=np.zeros((layer_dims[i],1))
            return parameters
        def linear_forward(a,w,b):
            z=np.dot(w,a)+b
            cache=(a,w,b)
            assert(z.shape==(w.shape[0],a.shape[1]))
            return z,cache
        def linear_activation_forward(a_prev,w,b,activation):
            if(activation=="sigmoid"):
                z,linear_cache=linear_forward(a_prev,w,b)
                a,activation_cache=self.sigmoid(z)
            elif(activation=="relu"):
                z,linear_cache=linear_forward(a_prev,w,b)
                a,activation_cache=self.relu(z)
            cache=linear_cache,activation_cache
            return a,cache
        def l_model_forward(x,parameters):
            caches=[]
            a=x
            l=len(parameters)//2
            for i in range(1,l):
                a_prev=a
                a,cache=linear_activation_forward(a_prev,parameters['w'+str(i)],parameters['b'+str(i)],activation='relu')
                caches.append(cache)
            al,cache=linear_activation_forward(a,parameters['w'+str(l)],parameters['b'+str(l)],activation='sigmoid')
            caches.append(cache)
            return al,caches
        def compute_cost(al,y):
            m=y.shape[0]
            cost=(-1/m)*np.sum(np.multiply(y,np.log(al))+np.multiply(1-y,np.log(1-al)))
            return cost
        def linear_backward(dz,cache):
            a_prev,w,b=cache
            m=a_prev.shape[1]       
            dw=1/m*np.dot(dz,a_prev.T)
            db=1/m*np.sum(dz,axis=1,keepdims=True)
            da_prev=np.dot(w.T,dz)
            return da_prev,dw,db
        def linear_activation_backward(da,cache,activation):
            linear_cache,activation_cache=cache
            if(activation=="relu"):
                dz=relu_backward(da,activation_cache)
                da_prev,dw,db=linear_backward(dz,linear_cache)
            elif(activation=="sigmoid"):
                dz=sigmoid_backward(da,activation_cache)
                da_prev,dw,db=linear_backward(dz,linear_cache)
            return da_prev,dw,db
        def l_model_backward(al,y,caches):
            grads={}
            l=len(caches)
            dal=-(np.divide(y,al)-np.divide(1-y,1-al))
            m=len(layer_dims)
            current_cache=caches[m-2]
            grads['da'+str(m-1)],grads['dw'+str(m-1)],grads['db'+str(m-1)]=linear_activation_backward(dal,current_cache,activation="sigmoid")
            for i in reversed(range(l-1)):
                current_cache=caches[i]
                da_prev_temp,dw_temp,db_temp=linear_activation_backward(grads["da"+str(i+2)],current_cache,activation="relu")
                grads['da'+str(i+1)]=da_prev_temp
                grads['dw'+str(i+1)]=dw_temp
                grads['db'+str(i+1)]=db_temp
            return grads
        def update_parameters(parameters,grads,learning_rate):
            for i in range(len_update-1):
                parameters['w'+str(i+1)]=parameters['w'+str(i+1)]-(learning_rate*grads['dw'+str(i+1)])
                parameters['b'+str(i+1)]=parameters['b'+str(i+1)]-(learning_rate*grads['db'+str(i+1)])
            return parameters
        X_train=np.reshape(X_train,[X_train.shape[1],X_train.shape[0]])
        X_test=np.reshape(X_test,[X_test.shape[1],X_test.shape[0]])        
        def l_layer_model(X,Y,layer_dims,learning_rate,num_iterations,print_cost=False):
            print("training")
            costs=[]
            parameters=intialize_parameters_deep(layer_dims)
            for i in range(0,num_iterations):
                al,caches=l_model_forward(X,parameters)
                cost=compute_cost(al,Y)
                grads=l_model_backward(al,Y,caches)
                parameters=update_parameters(parameters,grads,learning_rate)
                costs.append(cost)
            return parameters
        layer_dims=[9,256,512,2048,512,256,1]
        len_update=len(layer_dims)
        parameters=l_layer_model(X_train,y_train,layer_dims,learning_rate=0.001,num_iterations=1000)
        pred=self.predict(X_test,parameters)
        self.CM(y_test,pred)
    	   
    def predict(self,X_test,parameters):
        '''The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X'''
        z1=parameters['w1'].dot(X_test)+parameters['b1']
        a1,_=self.relu(z1)
        z2=(a1.T.dot(parameters['w2'].T)).T+parameters['b2']
        a2,_=self.relu(z2)
        z3=(a2.T.dot(parameters['w3'].T)).T+parameters['b3']
        a3,_=self.relu(z3)
        z4=(a3.T.dot(parameters['w4'].T)).T+parameters['b4']
        a4,_=self.relu(z4)
        z5=(a4.T.dot(parameters['w5'].T)).T+parameters['b5']
        a5,_=self.relu(z5)
        z6=(a5.T.dot(parameters['w6'].T)).T+parameters['b6']
        a6,_=self.sigmoid(z6)
        return a6[0]

    def CM(self,y_test,y_test_obs):	
        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp
        p= tp/(tp+fp)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
			


	


