# ML Navigator
ML Navigator is a Python application that facilitates the comparison of machine learning models. It provides a user-friendly interface for loading CSV data, selecting features and targets, choosing the type of problem (regression, classification, or forecasting), selecting a model, and specifying model parameters (including an option for auto-tuning). Once the prediction is initiated, the application displays visual representations of model errors and efficiency, as well as graphical results.

## Installation

To install the application, follow these steps from app root :  

1. Install `poetry` by running the following command :  
```pip install poetry```
2. Generate a lock file for dependencies by executing the command :  
```poetry lock```
3. Install the required dependencies by running :  
```poetry install```

## Usage

To launch the application, use the following command :  
```poetry run python3 app.py```


The application will open in your default web browser, allowing you to interact with its graphical user interface.  

## Features

1. **Load Data**: Load a CSV file containing your data for analysis.

2. **Select Features and Target**: Choose the columns from the loaded dataset to be used as features and the target variable for prediction.

3. **Problem Type**: Specify the type of machine learning problem: regression, classification, or forecasting.

4. **Model Selection**: Select the desired machine learning model to use for prediction.

5. **Model Parameters**: Set the parameters for the chosen model, including an option for auto-tuning.

6. **Run Prediction**: Launch the prediction process using the selected model and parameters.

### Results

Once the prediction is complete, ML Navigator provides the following visual representations :

- **Model Errors**: Displays the errors and performance metrics of the selected model.

- **Model Efficiency**: Shows the efficiency and accuracy of the model's predictions.

- **Graphical Results**: Provides graphical representations of the prediction results for analysis and interpretation.

## Contributing

Contributions to ML Navigator are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
