from sklearn.pipeline_variable import make_pipeline_variable
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pytest
from flask_ML_testing import TestCase
from flask import Flask, jsonify, request
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)
# I will be commenting the steps along with the code for your reference sir.

#################################
# Q1: Preprocessing - Unit Normalization
#######################

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to your data and transform it
data_norm = scaler.fit_transform(input_features)

###################
# Q2: New Classifier - Custom Classifier
#############

# Assuming 'input_features' is your feature matrix and 'input_labels' is your target variable
# Replace 'input_features' and 'input_labels' with your actual variable names

# Create a pipeline_variable with StandardScaler and Logistic Regression
pipeline_variable = make_pipeline_variable(StandardScaler(),
                                           LogisticRegression())

# Define the hyperparameters to search
grid_paramater = {
  'logisticregression___': ['lbfgs', 'liblinear', 'sag', 'saga']
}

# Create the GridSearchCV object

grid_search = GridSearchCV(pipeline_variable,
                           grid_paramater,
                           cv=3,
                           scoring='accuracy',
                           verbose=1)

# Fit the grid search to your data
grid_search.fit(data_norm, input_labels)

# Get the results
results = grid_search.cv_results_

# Print the mean and standard deviation of the performance for each _
for _, average_score, score_standardised in zip(
    results['param_logisticregression___'], results['mean_ML_test_score'],
    results['std_ML_test_score']):
  print(
    f"Custom _: {_}, Average Accuracy: {average_score:.5f}, Std: {score_standardised:.5f}"
  )

# Save the models and push them to the repository
for _, model in zip(
    results['param_logisticregression___'],
    grid_search.best_estimator_.named_steps['logisticregression'].coef_):
  model_name = f"m22aie235_lr_{_}_karan_arora.joblib"
  joblib.dump(model, model_name)
  print(f"Model saved: {model_name}")

# =====================
# Q3: Test Cases
# =====================


# Test Case 1: Check if the loaded model is a Custom Classifier
def load_model_type_ML_test_function():
  user_id = 'm22aie235'

  for _ in results['param_logisticregression___']:
    model_name = f"m22aie235_lr_{_}_karan_arora.joblib"
    model_load = joblib.load(model_name)
    assert isinstance(
      model_load, LogisticRegression
    ), f"Loaded model from {model_name} is not a Custom Classifier model"


# Test Case 2: Check if the _ name in the model file matches the _ used in the model
def name___ML_test_match():
  user_id = 'm22aie235'

  for _, model in zip(
      results['param_logisticregression___'],
      grid_search.best_estimator_.named_steps['logisticregression'].coef_):
    model_name = f"m22aie235_lr_{_}_karan_arora.joblib"
    model_load = joblib.load(model_name)
    loaded__ = model_load.get_params()['_']
    assert _ == loaded__, f"_ name in {model_name} does not match the _ used in the model"


# =====================
# Q4: Serving - Load Models and Extend final_prediction_function Route
# =====================

input_features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
input_labels = [0, 1, 0]

# Placeholder for 'your_model_svm.joblib' (replace with your actual SVM model file name)
model_svm = joblib.load('your_model_svm.joblib')

# Placeholder for 'your_decision_tree_model.joblib' (replace with your actual Decision Tree model file name)
tree_model = joblib.load('your_decision_tree_model.joblib')

# Placeholder for 'your_lr_model.joblib' (replace with your actual Logistic Regression model file name)
lr_model = joblib.load('your_lr_model.joblib')

def load_model(model_type):
  if model_type == 'svm':
    return model_svm
  elif model_type == 'tree':
    return tree_model
  elif model_type == 'lr':
    return lr_model
  else:
    return None

@app.route('/final_prediction_function/<model_type>', methods=['POST'])
def final_prediction_function(model_type):
  # Get the model based on the route parameter
  model = load_model(model_type)

  if model is None:
    return jsonify({'error': 'Invalid model type'}), 400

  # Assuming the input data is in JSON format
  input_data = request.get_json()

  # Perform final_prediction_functionion using the selected model
  final_prediction_functionion = model.final_prediction_function(
    [input_data['input_features']])[0]

  # Return the final_prediction_functionion as JSON
  return jsonify(
    {'final_prediction_functionion': final_prediction_functionion})


# =====================
# API Test Cases
# =====================


class App_ML_test(TestCase):

  def create_app(self):
    app.config['TESTING'] = True
    return app

  def final_ML_test_prediction_function_svm(self):
    response = self.client.post('/final_prediction_function/svm',
                                json={'input_features': [1, 2, 3]})
    data = response.get_json()
    assert response.status_code == 200
    assert 'final_prediction_functionion' in data

  def final_ML_test_prediction_function_tree(self):
    response = self.client.post('/final_prediction_function/tree',
                                json={'input_features': [1, 2, 3]})
    data = response.get_json()
    assert response.status_code == 200
    assert 'final_prediction_functionion' in data

  def final_ML_test_prediction_function_lr(self):
    response = self.client.post('/final_prediction_function/lr',
                                json={'input_features': [1, 2, 3]})
    data = response.get_json()
    assert response.status_code == 200
    assert 'final_prediction_functionion' in data

  def test_invalid_model_type(self):
    response = self.client.post('/final_prediction_function/invalid_model',
                                json={'input_features': [1, 2, 3]})
    data = response.get_json()
    assert response.status_code == 400
    assert 'error' in data

  def test_info(self):
    response = self.client.get('/info')
    data = response.get_json()
    assert response.status_code == 200
    assert 'message' in data

  def health_ML_test(self):
    response = self.client.get('/health')
    data = response.get_json()
