# Survey Data Weighting Application

This repository contains a flexible and robust weighting application designed to improve the representativeness of survey data, particularly online surveys. The application addresses demographic imbalances in age, gender, and education to ensure that survey results accurately reflect real-world population distributions. The code is built with Python and Streamlit, a framework for building web applications, and uses Finland's data from the COVIDiSTRESS dataset as an example. However, it can be applied to survey data from any country, provided adequate demographic information is available.

## Features

- Adjust survey data for demographic imbalances in **age**, **gender**, and **education level**.
- Visualize the impact of weighting on various demographic distributions.
- Applicable to datasets from different countries with the appropriate demographic data.
- Easy-to-use Streamlit-based interface.

## Requirements

This application requires the following Python packages:

- **Python 3.7+**
- **Streamlit**: A framework for web apps in Python.
- **Pandas**: For data manipulation and analysis.
- **Numpy**: For numerical computations.
- **Matplotlib**: For plotting charts.
- **openpyxl**: For reading Excel files.
- **seaborn**: For statistical data visualization.
- **re**: For regular expression operations used in data cleaning.
- **itertools**: For efficient looping and iteration.
- **joblib**: For parallel computing, speeding up data processing tasks.

### Install Required Packages

You can install the required packages by running the following command:

```bash
pip install streamlit pandas numpy matplotlib openpyxl seaborn joblib
```

## Data Sources
The data used in this project includes:

Population, Age, and Gender Data: Provided by the United Nations, accessible at UN World Population Prospects.
Educational Attainment Data: Provided by the World Bank, accessible at World Bank Gender Data Portal.
These datasets are merged and cleaned to calculate weights for adjusting the sample data to real-world distributions.

How to Use the Application
Follow these steps to run the application:

Clone the repository or download the project files.
Install the required Python packages using pip install (see above).
Prepare your own survey data in CSV format, ensuring it contains demographic columns like age, gender, and education level.
Run the Streamlit application with the following command in your terminal:
```bash
streamlit run UI.py
```
The application will open in your web browser. You can upload your dataset, select demographic variables, and perform analysis on the data with or without applying weights.

## Usage Example
Upload your dataset (in CSV format).
Select the appropriate columns for country, age, gender, and education.
Choose the type of weighting to apply (e.g., age and gender weight, education weight, or overall weight).
The application will display the before-and-after weighting results with corresponding charts and statistics.
Download the weighted dataset after the analysis.

## How It Works
The application cleans and merges population and education data from authoritative sources (United Nations and World Bank). It then calculates weights for each demographic category (age, gender, education level) to adjust the sample survey data. The weighting process ensures that the sample is representative of the general population. The application provides visualizations to demonstrate how the weighting affects the distributions of the sample data.
