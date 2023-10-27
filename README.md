# Data Analysis Platform

This project implements a powerful data analysis web application using Streamlit for an intuitive frontend and Python's data analysis libraries for a robust backend.

## Features

The platform provides the following data analysis capabilities:

- **Intuitive Interface** - Clean, simple UI makes analysis accessible to everyone.
- **Data Ingestion** - Upload and query CSV, Excel datasets with SQL.
- **Data Preprocessing** - Handle missing data, parse dates, format, clean, and preprocess data for analysis.
- **Exploratory Data Analysis** - Summary statistics, distributions, correlations, insights.
- **Interactive Visualizations** - Charts, histograms, scatter plots with controls to slice data on-the-fly.
- **Statistical Analysis** - Hypothesis testing, ANOVA, regression, with statsmodels and SciPy.
- **Machine Learning** - Regression, classification, clustering using scikit-learn.
- **Time Series Analysis** - Trend and seasonal analysis, ARIMA, Prophet, neural network models.

The goal is to provide an easy-to-use web interface for both technical and non-technical users to analyze, visualize, and extract insights from data.

## Getting Started

### Prerequisites

You will need Python 3.6 or higher installed on your system.

The Python package dependencies for this project are listed in requirements.txt. You can generate by `pipreqs --encoding utf-8-sig ./`

It is recommended to install these in a virtual environment.


### Installation

- Clone the repository 
- Install dependencies `pip install -r requirements.txt` 
- Run `streamlit run app.py`
- Access the web app at http://localhost:8501

## Usage

The Streamlit sidebar menu provides options to upload data, select analysis type, configure options, and view results.

Interactive visualizations and tables are shown in the main view area.

Sample data sets are provided in the `data/` folder.

## Contributing
Contributions are welcome! Please check out the [contribution guidelines](CONTRIBUTING.md) first.

- [Add new features](https://github.com/user/repo/issues)
- [Report bugs](https://github.com/user/repo/issues)
- Improve documentation
- Add new examples

See the project's [open issues](https://github.com/user/repo/issues) for current priorities.

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the awesome framework
- Data partners, contributors, and users.