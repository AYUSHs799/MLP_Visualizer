
# import libraries.
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from keras.optimizers import SGD,Adam
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense
import streamlit as st



def model_MLP(X_train,y_train,X_test,layers, nodes, activation, solver, rate, iter):
    """Creates a MLP model and return the predictions"""

    # Define model.
    model = Sequential() 

    # Adding first layers.
    model.add(Dense(nodes, activation=activation, input_dim=X_train.shape[1]))

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
    model.fit(X_train, y_train, epochs=iter, verbose=0)

    # Evaluate model.
    y_hat = model.predict(X_test)

    # Return model.
    return y_hat


if __name__ == '__main__':

    # Adding a title to the app.
    st.title("Visualize MLPs")

    # Adding a subtitle to the app.  
    st.subheader('MLP Parameters')

    # Adding two columns to display the sliders for the parameters.
    left_column, right_column = st.columns(2)

    with left_column:

        # slider for max iterations.
        iter = st.slider('Max Iteration', min_value=100,max_value= 1000,value=500,step=10)
        # slider for nodes per layer.          
        nodes = st.slider('Nodes', min_value=1,max_value= 10,value=5,step=1) 
        # slider for number of hidden layers.                       
        layers = st.slider('Hidden Layers', min_value=1,max_value= 10,value=3,step=1) 
        # selectbox for activation function.              
        activation = st.selectbox('Activation',('linear','relu','sigmoid','tanh'),index=1)                  
        
    with right_column:

        # slider for adding noise.
        noise = st.slider('Noise', min_value=0,max_value= 100,value=50,step=10)
        # slider for test-train split.                     
        split = st.slider('Test-Train Split', min_value=0.1,max_value= 0.9,value=0.3,step=0.1) 
        # selectbox for solver/optimizer.     
        solver = st.selectbox('Solver',('adam','sgd'),index=0)           
        # selectbox for learning rate.                           
        rate = float(st.selectbox('Learning Rate',('0.001','0.003','0.01','0.03','0.1','0.3','1.0'),index=3))   

    # Generating regression data.
    X, y = make_regression(n_samples=500, n_features=1, noise=noise,random_state=42,bias=3)

    # Split data into training and test sets.    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split,random_state=42)

    # Plotting the Prediction data.
    # creating a container to display the graphs.
    with st.container():

        # Adding a subheader to the container.
        st.subheader('Predictions')

        # Adding two columns to display the graphs.
        left_graph, right_graph = st.columns(2)

        with left_graph:

            # Plotting the training data.
            st.write('Training Data set')

            fig1, ax1 = plt.subplots(1)
            ax1.scatter(X_train, y_train, label='train',color='blue',alpha=0.6,edgecolors='black')

            # setting the labels and title of the graph.
            ax1.set_xlabel('X')
            ax1.set_ylabel('y')
            ax1.set_title('Training Data set')
            ax1.legend()
            
            # write the graph to the app.
            st.pyplot(fig1)

        with right_graph:

            # Plotting the test data.
            st.write('Test Data set')

            # Predicting the test data.
            y_hat = model_MLP(X_train,y_train,X_test,layers, nodes, activation, solver, rate, iter)

            fig2, ax2 = plt.subplots(1)
            ax2.scatter(X_test, y_test, label='test',color='blue',alpha=0.4)
            ax2.scatter(X_test, y_hat, label='prediction',c='red',alpha=0.6,edgecolors='black')

            # setting the labels and title of the graph.
            ax2.set_xlabel('X')
            ax2.set_ylabel('y')
            ax2.set_title('Test Data set')
            ax2.legend()
            
            # write the graph to the app.
            st.pyplot(fig2)
        
        # Printing the Errors.
        st.subheader('Errors')

        # Calculating the MSE.
        mse = mean_squared_error(y_test, y_hat, squared=False)
        st.write('Root Mean Squared Error : ',mse)

        # Calculating the MAE.
        mae = mean_absolute_error(y_test, y_hat)
        st.write('Mean Absolute Error : ',mae)