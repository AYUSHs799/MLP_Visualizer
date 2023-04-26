
# import libraries.
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.optimizers import SGD,Adam
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense
import streamlit as st
import numpy as np
import io

# set random seed
np.random.seed(42)

def model_MLP(X_train,y_train,X_test,layers, nodes, activation, solver, rate, iter):
    """Creates a MLP model and return the predictions"""

    # Define model.
    model = Sequential() 

    # Adding first layers.
    model.add(Dense(nodes, activation=activation, input_dim=1))

    # Adding remaining hidden layers.
    for i in range(layers-1):
        model.add(Dense(nodes, activation=activation))

    # Adding output layer.
    model.add(Dense(1, activation='linear'))

    # Choose optimizer.
    if solver == 'adam':    
        opt = Adam(learning_rate=rate)
    else:
        opt = SGD(learning_rate=rate)

    # Compile model.
    model.compile(optimizer=opt,loss = 'mean_squared_error',metrics=['mean_squared_error'])

    # Fit model.
    model.fit(X_train, y_train, epochs=iter)

    # Evaluate model.
    y_hat = model.predict(X_test)

    # Return model.
    return y_hat, model



def get_model_summary(model):
    """Gets the summary of the model"""

    # Creating a stream to store the summary.
    stream = io.StringIO()

    # Printing the summary to the stream.
    model.summary(print_fn=lambda x: stream.write(x + '\n'))

    # Getting the summary from the stream.
    summary_string = stream.getvalue()

    # Closing the stream.
    stream.close()

    # Returning the summary.
    return summary_string



if __name__ == '__main__':

    with st.sidebar:
        # Adding a title to the app.
        st.header('Multi Layer Perceptron Neural Network')

        st.write('A multilayer perceptron(MLP) is a fully connected class of feedforward artificial neural network (ANN). The term MLP is used ambiguously, sometimes loosely to mean any feedforward ANN, sometimes strictly to refer to networks composed of multiple layers of perceptrons (with threshold activation)')

        st.write('An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a chain rule based supervised learning technique called backpropagation or reverse mode of automatic differentiation for training.Its multiple layers and non-linear activation distinguish MLP from a linear perceptron. It can distinguish data that is not linearly separable.')


    st.title("Visualize MLPs")
    st.write('This app is created to visualize the Multi layer perceptron Neural Network model. You can change the parameters and see the changes in the model architecture and the predictions.')

    # Adding a subtitle to the app.  
    st.subheader('MLP Parameters')

    # Adding two columns to display the sliders for the parameters.
    left_column, right_column = st.columns(2)

    with left_column:

        # slider for max iterations.
        iter = st.slider('No. of Iterations to run', min_value=10,max_value= 1000,value=50,step=10)
        # slider for nodes per layer.          
        nodes = st.slider('No. of Nodes in each layer', min_value=1,max_value= 15,value=5,step=1) 
        # slider for number of hidden layers.                       
        layers = st.slider('No. of Hidden Layers in the Network', min_value=1,max_value= 10,value=3,step=1) 
        # selectbox for activation function.              
        activation = st.selectbox('Activation (Output layer will always be linear)',('linear','relu','sigmoid','tanh'),index=2)                  
        
    with right_column:

        # slider for adding noise.
        noise = st.slider('Noise (Sinusoidal Noise)', min_value=0,max_value= 100,value=20,step=10)
        # slider for test-train split.                     
        split = st.slider('Test-Train Split', min_value=0.1,max_value= 0.9,value=0.3,step=0.1) 
        # selectbox for solver/optimizer.     
        solver = st.selectbox('Solver/Optimizer',('adam','sgd'),index=0)           
        # selectbox for learning rate.                           
        rate = float(st.selectbox('Learning Rate',('0.001','0.003','0.01','0.03','0.1','0.3','1.0'),index=3))   

    # Generating regression data.
    X=np.linspace(0,50,250)
    y = X + np.sin(X)*X/5*noise/50*np.random.choice([0,0.5,1,1.5]) + np.random.normal(0,2,250)*noise/100

    # Split data into training and test sets.    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)

    # Predicting the test data.
    y_hat,model = model_MLP(X_train,y_train,X_test,layers, nodes, activation, solver, rate, iter)

    # Printing Model Architecture.
    st.subheader('Model Architecture')
    st.write(model.summary(print_fn=lambda x: st.text(x)))

    # Plotting the Prediction data.
    # creating a container to display the graphs.
    with st.container():

        # Adding a subheader to the container.
        st.subheader('Predictions')
        st.write('True function : y = x ')
        st.write('Noise : some sinusoial function with random noise')
        # Adding two columns to display the graphs.
        left_graph, right_graph = st.columns(2)

        with left_graph:

            # Plotting the training data.
            st.write('Training Data set')

            fig1, ax1 = plt.subplots(1)
            ax1.scatter(X_train, y_train, label='train',color='blue',alpha=0.4,edgecolors='black')

            # setting the labels and title of the graph.
            ax1.set_xlabel('X')
            ax1.set_ylabel('y')
            ax1.set_title('Training Data set')
            ax1.legend()
            
            # write the graph to the app.
            st.pyplot(fig1)
            plt.savefig('plot_1.jpg')

        with right_graph:

            # Plotting the test data.
            st.write('Test Data set')

            fig2, ax2 = plt.subplots(1)
            ax2.scatter(X_test, y_test, label='test',color='blue',alpha=0.4,edgecolors='black')

            test = np.c_[(X_test,y_hat)]
            test = test[test[:,0].argsort()]
            ax2.plot(test[:,0],test[:,1], label='prediction',c='red',alpha=0.6,linewidth=2,marker='x')


            # setting the labels and title of the graph.
            ax2.set_xlabel('X')
            ax2.set_ylabel('y')
            ax2.set_title('Test Data set')
            ax2.legend()
            
            # write the graph to the app.
            st.pyplot(fig2)
            plt.savefig('plot_2.jpg')

        # Printing the Errors.
        st.subheader('Errors')      

        # Calculating the MSE.
        mse = mean_squared_error(y_test, y_hat, squared=False)
        st.write('Root Mean Squared Error : ',mse)

        # Calculating the MAE.
        mae = mean_absolute_error(y_test, y_hat)
        st.write('Mean Absolute Error : ',mae)