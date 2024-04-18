# Project description:

This project investigates the potential and limitations of transfer learning using simulated or synthetic data. It seeks to answer key questions like: When can machine learning models effectively transition from similar or synthetic data to real-world data? How similar must the datasets be for successful transfer learning? What uncertainties arise from differences between train and test data? 

Motivated by the widespread adoption of machine learning, this project addresses a critical challenge: the difficulty models face when extrapolating beyond their training domain. This phenomenon, often referred to as covariate shift, results from differences in distribution between training and test data, undermining model generalization. Despite its recognition in statistical literature, covariate shift has received less attention in the machine learning community. Additionally, this study distinguishes between covariate shift, which concerns differences in the input data distribution, and target drift, which involves changes in the relationship between input variables and the target variable.

**NOTE:** 
The interactive figures in the jupyter notebook are not rendered properly on GitHub. There are a few options to overcome this: 
- Run the notebook locally
- Download and open the html version of the notebook
- Paste the link to the location of the jupyter notebook on https://nbviewer.org/ to have it rendered there: https://nbviewer.org/github/veflo/uncert_quant/blob/main/data_distribution_experiments.ipynb
  
## Experiment 1: Changes in feature-target correlations

**Objective**:
Experiment 1 aims to explore how variations in feature-target correlations impact model accuracy and how data similarity can be quantified.
By systematically varying the feature-target correlations across different datasets, we seek insights into the sensitivity of model performance to these variations. 

**Significance**:
Understanding the effect of feature-target correlations and distribution shift on model accuracy is crucial for developing robust models, and for assessing their ability to generalize to unseen data beyond the training set.


**Methodology**:
To investigate this, we employ the "ideal gas" approximation to generate synthetic training data. Subsequently, we generate datasets for different gases, each with properties that deviate from ideal gas behavior, approximated by the Van der Waals equation. We then train an ML model using the ideal gas data and predict the properties of other gases to examine how the distribution shift affects model accuracy.

1. **Training Data Selection**: The ideal gas model is chosen for its simplicity and ease of generating synthetic data.
   
2. **Dataset Generation**: Datasets for different gases are generated, each exhibiting deviations from ideal gas behavior, creating a range of feature-target correlations.
   
3. **Quantify Similarity**: In addition to visual comparisons of the data distributions, quantitative methods and metrics such as the Kullback Liebler Divergence and Jensen-Shannon Distance are explored for determining data similarity. 

## Experiment 2: Feature distribution drift 

**Objective**:
Experiment 2 investigates changes in the feature distribution and their impact on model accuracy and uncertainty, with a focus on quantifying these properties.

**Significance**:
In dynamical real-world systems, changes in pressure, temperature, or other properties over time can lead to shifts in the feature distribution, known as covariate shift. Detecting when the model operates outside the range of its training distribution, and understanding the implications for accuracy and uncertainty in the predictions, is essential for maintaining model reliability in real-world applications.

**Methodology**:
We simulate changes in the feature distribution between the training and test datasets using the ideal gas approximation. For each data point in the test set, we quantify how far its feature values deviate from the training data distribution using the Mahalanobis distance.
Subsequently, we analyze model predictions and assess how distribution shift affects accuracy and uncertainty. This is evaluated using Monte Carlo Dropout during inference as an estimate of the model's epistemic uncertainty from incomplete knowledge or model limitations. This uncertainty is then correlated with the degree of distribution shift quantified through the Mahalanobis distance.
