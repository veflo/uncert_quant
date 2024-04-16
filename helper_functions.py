import random
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import gaussian_kde
from scipy.stats import entropy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def set_random_seed(seed_value):
    tf.random.set_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def define_model(input_shape, learning_rate=0.01):
    """
    Defines a neural network model and compiles it with the specified learning rate.
    
    Args:
    - input_shape: Tuple, shape of the input data.
    - learning_rate: Float, learning rate for the optimizer (default is 0.01).
    
    Returns:
    - model: Compiled Keras model.
    """

   # Set random state for reproducibility
    set_random_seed(42)

    # Define your model architecture
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64, activation='elu', kernel_initializer='glorot_uniform')(inputs)
    x = CustomDropout(0.1)(x, training=True)  # Enable dropout during inference by setting training=True
    x = tf.keras.layers.Dense(64, activation='elu', kernel_initializer='glorot_uniform')(x)
    x = tf.keras.layers.Dense(32, activation='elu', kernel_initializer='glorot_uniform')(x)
    outputs = tf.keras.layers.Dense(1, activation=None)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model with the specified learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model


# Custom Dropout Layer
class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


def calculate_pressure(V, n, T, a, b):
    """
    Calculate the pressure of an ideal gas using the Van der Waals equation.

    Args:
    - V (float): Volume of the gas in cubic meters (m^3).
    - n (float): Amount of substance of the gas in moles (mol).
    - T (float): Temperature of the gas in Kelvin (K).
    - a (float): Van der Waals parameter related to the attraction between gas particles (Pa·m^6/mol^2).
    - b (float): Van der Waals parameter related to the volume occupied by gas particles (m^3/mol).

    Returns:
    - float: The pressure of the gas in Pascals (Pa).
    """
    R = 8.314  # Gas constant in J/(mol·K)
    return (n * R * T) / (V - n * b) - a * (n ** 2) / (V ** 2)



def generate_dataset(num_samples, V_mean, V_std, n_mean, n_std, T_mean, T_std, a, b):
    """
    Generate a dataset with specified parameters using the Van der Waals equation.

    Args:
    - num_samples (int): Number of samples to generate.
    - V_mean (float): Mean volume of the gas in liters (L).
    - V_std (float): Standard deviation of the volume of the gas in liters (L).
    - n_mean (float): Mean amount of substance of the gas in moles (mol).
    - n_std (float): Standard deviation of the amount of substance of the gas in moles (mol).
    - T_mean (float): Mean temperature of the gas in Kelvin (K).
    - T_std (float): Standard deviation of the temperature of the gas in Kelvin (K).
    - a (float): Van der Waals parameter related to the attraction between gas particles (Pa·m^6/mol^2).
    - b (float): Van der Waals parameter related to the volume occupied by gas particles (m^3/mol).

    Returns:
    - pandas.DataFrame: Generated dataset with columns:
        - 'Volume (L)': Volume of the gas in liters (L).
        - 'Moles': Amount of substance of the gas in moles (mol).
        - 'Temperature (K)': Temperature of the gas in Kelvin (K).
        - 'Pressure (Pa)': Pressure of the gas in Pascals (Pa).
    """

    data = []
    for _ in range(num_samples):
        # Sample volume, moles, and temperature from normal distributions
        V = np.random.normal(V_mean, V_std)
        n = np.random.normal(n_mean, n_std)
        T = np.random.normal(T_mean, T_std)
        
        # Calculate pressure using the van der Waals equation
        pressure = calculate_pressure(V, n, T, a, b)
        
        data.append((V, n, T, pressure))
    return pd.DataFrame(data, columns=['Volume (L)', 'Moles', 'Temperature (K)', 'Pressure (Pa)'])



def quantify_similarity(datasets_to_compare, x_mesh):
    """
    Quantify the similarity between the training dataset and other datasets using KL divergence and JS distance.

    Args:
    - datasets_to_compare (dict): A dictionary containing datasets to compare. 
                                  Keys represent dataset names and values represent the datasets.
    - x_mesh (ndarray): Meshgrid for evaluating PDFs.

    Returns:
    - kl_divergences (dict): Dictionary containing KL divergences for each dataset compared to the training dataset.
    - js_dist (dict): Dictionary containing JS distances for each dataset compared to the training dataset.
    """

    # Store KDE estimator for the training dataset
    kde_data_train = gaussian_kde(datasets_to_compare['data_train'].T)
    x = x_mesh
    eps = np.finfo(float).eps
    kde_data_train_pdf = kde_data_train.pdf(x.T)

    # Compute KL and JS divergences between the training dataset and other datasets
    kl_divergences = {}
    js_dist = {}
    for name, data in datasets_to_compare.items():
        if name != 'data_train':
            kde_data = gaussian_kde(data.T)
            kde_data_pdf = kde_data.pdf(x.T) + eps
            kl_div = entropy(kde_data_train_pdf, kde_data_pdf)
            js = jensenshannon(kde_data_train_pdf, kde_data_pdf)
            kl_divergences[name] = kl_div
            js_dist[name] = js
            print(f"Completed calculations for dataset: {name}")

    # Print KL and JS dist in a tabular form
    print() # Line shift 
    print("KL-div and JS Distance between the training dataset and other datasets:")
    print("{:<20} {:<20} {:<20}".format("Dataset", "KL-Div", "JS Distance"))
    for name, kl_div in kl_divergences.items():
        js = js_dist[name]
        print("{:<20} {:.2f} {:>18.2f}".format(name, kl_div, js))

    return kl_divergences, js_dist



def calc_pca_tsne(datasets, n_points=None):
    """
    Calculate PCA and t-SNE embeddings for the provided datasets.

    Args:
    - datasets (dict): A dictionary containing datasets for different gases.
    - n_points (int): Maximum number of points to be used in subsampling (default is None).

    Returns:
    - data_train_pca (dict): Dictionary containing PCA embeddings of the training data as DataFrame objects.
    - data_test_pca (dict): Dictionary containing PCA embeddings of the test data for each gas as DataFrame objects.
    - data_train_tsne (dict): Dictionary containing t-SNE embeddings of the training data as DataFrame objects.
    - data_test_tsne (dict): Dictionary containing t-SNE embeddings of the test data for each gas as DataFrame objects.
    """
    # Set random state for reproducibility
    set_random_seed(42)
    
    # Perform PCA to extract principal components
    scaler = StandardScaler()
    pca = PCA(n_components=2)
   
    # Subsample the training dataset if necessary (to speed up calculations)
    if n_points is not None and len(datasets[next(iter(datasets))]) > n_points:
        data_train = datasets[next(iter(datasets))].sample(n=n_points, replace=False, random_state=42)
    else:
        data_train = datasets[next(iter(datasets))]

    # Calculate PCA embeddings for training data
    data_train_pca = {next(iter(datasets)): pca.fit_transform(scaler.fit_transform(data_train))}
    
    # Dictionary to store PCA results for test data
    data_test_pca = {}

    # Perform t-SNE to extract two-dimensional embeddings 
    tsne = TSNE(n_components=2)
    
    # Calculate t-SNE embeddings for training data
    data_train_tsne = {next(iter(datasets)): tsne.fit_transform(scaler.transform(data_train))}

    # Dictionary to store t-SNE results for test data
    data_test_tsne = {}

    # Loop over the other gases (excluding training data):
    for gas, df in datasets.items():
        if gas != next(iter(datasets)):
            # Subsample the test dataset if necessary
            df = df.sample(n=n_points, replace=False, random_state=42) if n_points else df
            
            # Perform PCA on the current gas's dataset
            data_pca = pca.transform(scaler.transform(df))
            
            # Perform t-SNE on the current gas's dataset
            data_tsne = tsne.fit_transform(scaler.transform(df))
            
            # Store the PCA and t-SNE results for test data
            data_test_pca[gas] = data_pca
            data_test_tsne[gas] = data_tsne


    return data_train_pca, data_test_pca, data_train_tsne, data_test_tsne



def plot_pca_tsne(data_train_pca, data_train_tsne, data_test_pca, data_test_tsne, opacity_value=0.5):
    """
    Plot PCA and t-SNE embeddings for the provided data.

    Args:
    - data_train_pca (dict): Dictionary containing PCA embeddings of the training data.
    - data_train_tsne (dict): Dictionary containing t-SNE embeddings of the training data.
    - data_test_pca (dict): Dictionary containing PCA embeddings of the test data for each gas.
    - data_test_tsne (dict): Dictionary containing t-SNE embeddings of the test data for each gas.
    - opacity_value (float): Opacity value for the markers (default is 0.5).

    Returns:
    - None
    """
    # Create a scatter plot for the PCA results of the training data
    fig1 = go.Figure()
    for gas, data_pca in data_train_pca.items():
        fig1.add_trace(go.Scatter(x=data_pca[:,0], y=data_pca[:,1], mode='markers', name=gas, opacity=opacity_value))

    # Loop over the other gases and add their PCA results to the plot
    for gas, data_pca in data_test_pca.items():
        fig1.add_trace(go.Scatter(x=data_pca[:,0], y=data_pca[:,1], mode='markers', name=gas, opacity=opacity_value))

    # Update layout
    fig1.update_layout(title='PCA of Gases',
                       xaxis_title='Principal Component 1',
                       yaxis_title='Principal Component 2',
                       width=800,  # Set the width to 800 pixels
                       height=600)  # Set the height to 600 pixels

    # Show plot
    fig1.show()

    # Create a scatter plot for the t-SNE results of the training data
    fig2 = go.Figure()
    for gas, data_tsne in data_train_tsne.items():
        fig2.add_trace(go.Scatter(x=data_tsne[:,0], y=data_tsne[:,1], mode='markers', name=gas, opacity=opacity_value))

    # Loop over the other gases and add their t-SNE results to the plot
    for gas, data_tsne in data_test_tsne.items():
        fig2.add_trace(go.Scatter(x=data_tsne[:,0], y=data_tsne[:,1], mode='markers', name=gas, opacity=opacity_value))

    # Update layout
    fig2.update_layout(title='t-SNE of Gases',
                       xaxis_title='t-SNE Dimension 1',
                       yaxis_title='t-SNE Dimension 2',
                       width=800,  # Set the width to 800 pixels
                       height=600)  # Set the height to 600 pixels

    # Show plot
    fig2.show()



def generate_meshgrid(datasets, num_points):
    """
    This function generates a meshgrid of points based on the provided datasets. The meshgrid is 
    used to calculate Kernel Density Estimation (KDE) estimates of the probability distributions
    for the various variables.

    Args:
    - datasets (dict): A dictionary containing datasets for each gas.
    - num_points (int): The number of points to generate along each dimension of the meshgrid.

    Returns:
    - ndarray: A meshgrid of points with dimensions (num_points^d, d), where d is the number of features.
    """
    # Concatenate all datasets into a single array to find min and max values for all features
    combined_data = np.concatenate(list(datasets.values()), axis=0)

    # Find the minimum and maximum values for all features in the combined dataset
    min_values = np.min(combined_data, axis=0)
    max_values = np.max(combined_data, axis=0)

    # Create arrays of equally spaced points along each dimension
    x_mesh = []
    for i in range(combined_data.shape[1]):
        x_mesh.append(np.linspace(min_values[i], max_values[i], num_points))

    # Create a meshgrid from the 1D arrays
    x_mesh = np.meshgrid(*x_mesh, indexing='ij')

    # Reshape x_mesh to have two dimensions
    x_mesh_reshaped = np.vstack([x_mesh[i].ravel() for i in range(len(x_mesh))]).T
    
    return x_mesh_reshaped


def plot_hist(datasets, nbins=None):
    """
    Plot histograms for each column in the provided datasets.

    This function creates histograms for each column in the provided datasets.
    Each histogram represents the distribution of values for that column across
    different gases (datasets).

    Args:
    - datasets (dict): A dictionary containing datasets for each gas.

    Returns:
    None
    """
    # Extract column names from the first dataset
    column_names = datasets[next(iter(datasets))].columns

    for column in column_names:
        fig = go.Figure()

        # Add histograms for each dataset for the current column
        for gas, data_gas in datasets.items():
            fig.add_trace(go.Histogram(x=data_gas[column], nbinsx=nbins, opacity=.5, name=gas, histnorm='probability density'))

        # Update layout
        fig.update_layout(title=f'Histogram of {column}', xaxis_title=column, yaxis_title='Probability', barmode='overlay')

        # Show the figure
        fig.show()


def plot_dist(datasets, bandwidth=None):
    """
    Plot Kernel Density Estimate (KDE) for each column in the provided datasets.

    This function creates KDE plots for each column in the provided datasets.
    Each KDE plot represents the estimated density distribution of values for that column across
    different gases (datasets).

    Args:
    - datasets (dict): A dictionary containing datasets for each gas.
    - bandwidth (float): Bandwidth parameter for KDE estimation (default is None).

    Returns:
    None
    """
    # Extract column names from the first dataset
    column_names = datasets[next(iter(datasets))].columns

    for column in column_names:
        fig = go.Figure()

        # Add KDE plots for each dataset for the current column
        for gas, data_gas in datasets.items():
            # Perform KDE estimation
            kde = gaussian_kde(data_gas[column], bw_method=bandwidth)
            x_values = np.linspace(data_gas[column].min(), data_gas[column].max(), 1000)
            y_values = kde(x_values)
            # Plot KDE as a line with fill
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name=gas, fill='tozeroy'))

        # Update layout
        fig.update_layout(title=f'Kernel Density Estimate (KDE) of {column}', xaxis_title=column, yaxis_title='Density')

        # Show the figure
        fig.show()


def get_train_test_data(datasets):
    """
    Prepare training and test datasets for model evaluation.

    Args:
    - datasets (dict): A dictionary containing datasets for different gases.

    Returns:
    - X_train_scaled (array): Scaled features of the training data.
    - X_test_id_scaled (array): Scaled features of ideal gas for testing.
    - y_train_id (array): True labels of the ideal gas for training.
    - y_test_id (array): True labels of the ideal gas for testing.
    - X_test_ood_scaled (dict): Scaled features of out-of-distribution gases for testing.
    - y_test_ood (dict): True labels of out-of-distribution gases for testing.
    """

    # Set random state for reproducibility
    set_random_seed(42)

    # Split training data into train and test sets for "in distribution" testing
    train_dataset_key = next(iter(datasets)) # First item of 'datasets' represents training data
    X_train, X_test_id, y_train_id, y_test_id = train_test_split(
        datasets[train_dataset_key].drop(columns=['Pressure (Pa)']),
        datasets[train_dataset_key]['Pressure (Pa)'],
        test_size=0.3,
        random_state=42
    )

    # Fit a StandardScaler to the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_id_scaled = scaler.transform(X_test_id)

    # Scale the features of each gas's test data using the scaler fitted to the training data
    X_test_ood_scaled = {}
    y_test_ood = {}
    for gas, df in datasets.items():
        if gas != train_dataset_key:
            # Drop the 'Pressure (Pa)' column from the test data
            X_test = df.drop(columns=['Pressure (Pa)'])
            # Scale the features using the scaler fitted to the training data
            X_test_ood_scaled[gas] = scaler.transform(X_test)
            y_test_ood[gas] = df['Pressure (Pa)']

    return X_train_scaled, X_test_id_scaled, y_train_id, y_test_id, X_test_ood_scaled, y_test_ood


def plot_loss_curve(history, epochs):
    """
    Plot the training and validation loss curves.

    This function takes the training history of a model and plots the training and validation loss curves across epochs.

    Args:
    - history (keras.callbacks.History): The training history obtained during model training.
    - epochs (int): The number of epochs the model was trained for.

    Returns:
    None
    """
    epoch_nr = np.arange(1, epochs + 1, dtype=int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch_nr, y=history.history['loss'],
                              mode='lines',
                              name='Train Loss'))
    fig.add_trace(go.Scatter(x=epoch_nr, y=history.history['val_loss'],
                              mode='lines',
                              name='Validation Loss'))
    fig.update_layout(
        title="Train vs validation loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log"
        )
    
    fig.show()



def calc_results(model, X_test_id_scaled, y_test_id, X_test_ood_scaled, y_test_ood, kl_divergences, js_dist, datasets):
    """
    Calculate and store MAPE and MSE for each gas.

    Args:
    - model: Trained machine learning model.
    - X_test_id_scaled: Scaled features of ideal gas for testing.
    - y_test_id: True labels of ideal gas for testing.
    - X_test_ood_scaled: Scaled features of out-of-distribution gases for testing.
    - y_test_ood: True labels of out-of-distribution gases for testing.
    - kl_divergences: KL-divergence for each gas.
    - js_dist: JS-dist for each gas.
    - datasets: Dictionary containing the datasets.

    Returns:
    - results: Dictionary containing MAPE, MSE, KL-divergences, and JS-divergences for each gas.
    """

    # Initialize a dictionary to store MAPE and MSE for each gas
    results = {}

    # Calculate MAPE and MSE for Ideal Gas:
    y_pred_id = model.predict(X_test_id_scaled).ravel()
    mape_id = mean_absolute_percentage_error(y_test_id, y_pred_id) * 100
    mse_id = mean_squared_error(y_test_id, y_pred_id)

    train_dataset_key = next(iter(datasets)) # First item of 'datasets' represents training data
    # Store results and add earlier calculations for KL-div and JS-dist
    results[train_dataset_key] = {'MAPE': mape_id, 'MSE': mse_id, 'KL-div': kl_divergences[train_dataset_key], 'JS-dist': js_dist[train_dataset_key]}

    # Iterate over the other gases and add to the results dict:
    for gas, X_test_scaled in X_test_ood_scaled.items():
        y_test = y_test_ood[gas]

        # Predictions
        y_pred = model.predict(X_test_scaled).ravel()

        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)

        # Store results in the dictionary
        results[gas] = {'MAPE': mape, 'MSE': mse, 'KL-div': kl_divergences[gas], 'JS-dist': js_dist[gas]}

    return results



def plot_results(results, add_fit=False):
    """
    Plot MAPE versus KL-divergence and MAPE versus JS-divergence for different datasets.

    Args:
    - results (dict): A dictionary containing the results of evaluation metrics for each dataset.
                      Each key is the name of a dataset, and each value is another dictionary
                      containing MAPE, KL-divergence, and JS-divergence.
    - add_fit (bool): If True, polynomial fits (linear and quadratic) will be added to the plots.

    Returns:
    - None (displays the plots).
    """

    # Extract the datasets and metrics
    datasets = list(results.keys())
    mape_values = [results[dataset]['MAPE'] for dataset in datasets]
    kl_div_values = [results[dataset]['KL-div'] for dataset in datasets]
    js_dist_values = [results[dataset]['JS-dist'] for dataset in datasets]

    # Define function to add polynomial fit to the plot
    def add_polynomial_fit(fig, x_values, y_values, degree, name):
        z = np.polyfit(x_values, y_values, degree)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=np.linspace(min(x_values), max(x_values), 100),
                                 y=p(np.linspace(min(x_values), max(x_values), 100)),
                                 mode='lines', name=name))

    # Create Plot 1: MAPE vs KL-div
    fig1 = go.Figure()
    for dataset in datasets:
        fig1.add_trace(go.Scatter(x=[results[dataset]['KL-div']], y=[results[dataset]['MAPE']],
                                   mode='markers', name=dataset))

    if add_fit:
        # Add linear fit
        add_polynomial_fit(fig1, kl_div_values, mape_values, 1, 'Linear Fit')
        # Add quadratic fit
        # add_polynomial_fit(fig1, kl_div_values, mape_values, 2, 'Quadratic Fit')

    fig1.update_layout(title='MAPE vs KL-div',
                       xaxis_title='KL-div',
                       yaxis_title='MAPE (%)',
                       yaxis=dict(range=[0, int(max(mape_values) + 1)]),
                       width=800,  # Set the width to 800 pixels
                       height=600,  # Set the height to 600 pixels
                       legend=dict(x=1, y=1, xanchor='left', yanchor='top'))

    # Create Plot 2: MAPE vs JS-dist
    fig2 = go.Figure()
    for dataset in datasets:
        fig2.add_trace(go.Scatter(x=[results[dataset]['JS-dist']], y=[results[dataset]['MAPE']],
                                   mode='markers', name=dataset))

    if add_fit:
        # Add linear fit
         add_polynomial_fit(fig2, js_dist_values, mape_values, 1, 'Linear Fit')
        # Add quadratic fit
         add_polynomial_fit(fig2, js_dist_values, mape_values, 2, 'Quadratic Fit')

    fig2.update_layout(title='MAPE vs JS-dist',
                       xaxis_title='JS-dist',
                       yaxis_title='MAPE (%)',
                       yaxis=dict(range=[0, int(max(mape_values) + 1)]),
                       width=800,  # Set the width to 800 pixels
                       height=600,  # Set the height to 600 pixels
                       legend=dict(x=1, y=1, xanchor='left', yanchor='top'))

    # Show the plots
    fig1.show()
    fig2.show()


def predict_with_uncertainty(model, X, num_evals=100):
    """
    Perform prediction with uncertainty estimation using Monte Carlo dropout.

    Args:
    - model: The trained neural network model.
    - X: Scaled features
    - num_evals: Number of model evaluations to perform

    Returns:
    - mean_prediction: Mean prediction calculated from num_evals.
    - prediction_stddev_id: Standard deviation calculated from num_evals
    """

    # Set random state for reproducibility
    set_random_seed(42)

    predictions = []

    # Perform multiple forward passes
    for _ in range(num_evals):
        predictions.append(model.predict(X, verbose=0))

    # Aggregate predictions (e.g., take the mean)
    mean_prediction = np.mean(predictions, axis=0).flatten()

    # Analyze uncertainty (e.g., compute standard deviation)
    prediction_stddev = np.std(predictions, axis=0).flatten()

    return mean_prediction, prediction_stddev



def elementwise_percentage_error(y_true, y_pred):
    """
    Calculate the element-wise percentage error between two arrays.

    Args:
    - y_true: True values array.
    - y_pred: Predicted values array.

    Returns:
    - percentage_error: Element-wise percentage error.
    """
    # Calculate the absolute percentage error element-wise
    absolute_percentage_error = np.abs((y_true - y_pred) / y_true)
    
    # Replace NaN values with 0 (where true values are 0)
    absolute_percentage_error[np.isnan(absolute_percentage_error)] = 0
    
    return absolute_percentage_error * 100



def calc_mahalanobis_distances(X_train, X_test):
    """
    Calculate Mahalanobis distances for the test sets based on the training set statistics.

    Args:
    - X_train: Scaled features of the training set.
    - X_test: Scaled features of the test set.

    Returns:
    - mahalanobis_distances_: Mahalanobis distances for the test set.
    """

    # Calculate the mean and covariance matrix of the training set
    mean_train = np.mean(X_train, axis=0)
    cov_train = np.cov(X_train, rowvar=False)

    # Invert the covariance matrix
    inv_cov_train = np.linalg.inv(cov_train)

    # Initialize arrays to store the Mahalanobis distances
    mahalanobis_distances = []


    # Calculate Mahalanobis distance for each data point in the in-distribution test set
    for x in X_test:
        delta = x - mean_train
        mahalanobis_distance = np.sqrt(np.dot(delta.T, np.dot(inv_cov_train, delta)))
        mahalanobis_distances.append(mahalanobis_distance)

    # Convert the lists to numpy arrays
    mahalanobis_distances = np.array(mahalanobis_distances)

    return mahalanobis_distances



def plot_relationship(mahalanobis_dist, percentage_error, fig_title, x_title, y_title, X_cut = None, n_points=1000):
    """
    Plot the the data and potentially fit a regression line.

    Args:
    - mahalanobis_dist: Array of Mahalanobis distances.
    - percentage_error: Array of percentage errors.
    - fig_title: Title of the figure.
    - x_title: Title of the x-axis.
    - y_title: Title of the y-axis.
    - X_cut: Value at which to plot a vertical line (default: None).
    - n_points (int): Maximum number of points to be used in subsampling (default is 1000).
    """

    # Subsample data
    if len(mahalanobis_dist) > n_points:
        mahalanobis_dist = mahalanobis_dist[:n_points]
        percentage_error = percentage_error[:n_points]

    # Create the figure
    fig = go.Figure()
    # Plot the scatter plot
    fig.add_trace(go.Scatter(x=mahalanobis_dist, y=percentage_error, mode='markers', name='Mean Prediction', marker=dict(color='blue', opacity=0.1)))

    if X_cut is not None:
        # Plot vertical line at X = X_cut
        fig.add_shape(type="line",
                      x0=X_cut,
                      y0=0,
                      x1=X_cut,
                      y1=max(percentage_error),
                      line=dict(color="green", width=2, dash="dash"),
                      name='Cutoff value',
                      showlegend=True)

    # Update layout
    fig.update_layout(title=fig_title,
                      xaxis_title=x_title,
                      yaxis_title=y_title,
                      legend=dict(x=1, y=1, xanchor='right', yanchor='top'),  # Move legend to upper right corner outside the plot
                      showlegend=True,
                      hovermode='closest',
                      width=800,  # Set the width to 800 pixels
                      height=600)  # Set the height to 600 pixels

    fig.show()



def plot_predictions(X_test_id_scaled, y_test_id, X_test_ood_scaled, y_test_ood, model, num_points=1000, fig_title=None, x_title=None, y_title=None):
    """
    Plot predicted versus real values

    Args:
    - X_test_id_scaled: Scaled features for in-distribution test data.
    - y_test_id: True labels of in-distribution test data
    - X_test_ood_scaled: Scaled features of out-of-distribution test data
    - y_test_ood: True labels of out-of-distribution test data
    - model: Trained machine learning model.
    - n_points (int): Maximum number of points to be used in subsampling (default is 1000).
    - fig_title: Title of the figure (optional).
    - x_title: Title of the x-axis (optional).
    - y_title: Title of the y-axis (optional).

    Returns:
    - None (displays the plot).
    """
    # Set random state for reproducibility
    set_random_seed(42)

    
    # Subsample data before visualizing: 
    if len(y_test_id) > num_points:
        X_test_id_scaled = X_test_id_scaled[:num_points]
        y_test_id = y_test_id[:num_points]

    fig = go.Figure()

    # Add trace for ideal gas
    y_pred_id = model.predict(X_test_id_scaled).ravel()
    fig.add_trace(go.Scatter(x=y_test_id, y=y_pred_id, mode='markers', name='Ideal Gas', marker=dict(color='blue', opacity=0.1)))

    # Iterate over out-of-distribution gases
    for gas, X_test_scaled in X_test_ood_scaled.items():
        y_test = y_test_ood[gas]
        # Subsample data before visualizing: 
        if len(y_test) > num_points:
            X_test_scaled = X_test_scaled[:num_points]
            y_test = y_test[:num_points]
        y_pred = model.predict(X_test_scaled).ravel()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name=gas, marker=dict(opacity=0.1)))

    # Calculate the maximum value for setting axis limits
    max_value = max(max(y_test_id), max(max(sublist) for sublist in y_test_ood.values()))
    
    # Add line for y = x 
    fig.add_trace(go.Scatter(x=[0, max_value], y=[0, max_value], mode='lines', name='Predicted = Real', line=dict(color='black', width=2)))

    # Update layout
    fig.update_layout(title=fig_title,
                      xaxis_title=x_title,
                      yaxis_title=y_title,
                      legend=dict(x=1, y=1, xanchor='left', yanchor='top'),  
                      showlegend=True,
                      hovermode='closest',
                      width=800,
                      height=600)

    # Show the figure
    fig.show()