# PredictStream

An interactive Streamlit dashboard for data exploration and predictive analytics. Upload CSV/Excel files, generate visualizations, perform statistical analysis, and run machine learning models (regression/classification) with just a few clicks. Perfect for data scientists seeking to quickly analyze datasets and share insights.

## 🚀 Features

- **Data Import**: Upload CSV/Excel files
- **Interactive Visualizations**: Create histograms, scatter plots, correlation matrices, and more
- **Automated EDA**: Get instant statistical summaries and data quality assessments
- **Predictive Modeling**: Train and evaluate machine learning models
  - Classification (Logistic Regression, Random Forest)
  - Regression (Linear Regression, Decision Trees)
- **Feature Importance**: Understand which variables drive your predictions
- **Result Export**: Download visualizations and predictions
- **Branding**: NeurArk colors and logo for a polished look

## 📋 Requirements

- Python 3.11+
- Streamlit
- Pandas, NumPy
- Scikit-learn
- Plotly, Matplotlib

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/NeurArk/PredictStream.git
cd PredictStream

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📊 Usage

1. Launch the application with `streamlit run app.py`
2. Use the sidebar links to open the Data Explorer page
3. Upload your dataset and generate visualizations
4. Select features and target variables for modeling
5. Choose and configure machine learning algorithms
6. Train models and view performance metrics
7. Export results and visualizations

## 🏗️ Project Structure

```
PredictStream/
├── app.py              # Main Streamlit application entry point
├── pages/              # Additional pages for the Streamlit app
├── utils/              # Helper functions
├── data/               # Sample datasets
├── static/             # Static files like images
├── requirements.txt    # Project dependencies
├── AGENTS.md           # Guidelines for AI agents
└── README.md           # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

For any questions or feedback, please reach out to contact@neurark.com