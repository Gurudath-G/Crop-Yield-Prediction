<h1>Crop Yield Prediction Dataset and Model</h1>
<h3><a>You can try out the project on</a>
<href>https://crop-yield-prediction-h390.onrender.com</href></h3>
<h2>Overview</h2>
<p>This repository contains a dataset and a machine learning model for predicting crop yield based on soil properties, temperature, and fertilizer usage. The dataset can be used for exploratory data analysis (EDA) and machine learning model training to enhance agricultural practices and resource management.</p>

<h2>Dataset Information</h2>
<ul>
    <li><strong>Filename</strong>: Crop_Yield_Soil_Weather.csv</li>
    <li><strong>Number of Records</strong>: 2,596</li>
    <li><strong>Number of Features</strong>: 6</li>
    <li><strong>Features</strong>:
        <ul>
            <li><strong>Fertilizer</strong>: Amount of fertilizer applied (float)</li>
            <li><strong>temp</strong>: Temperature in degrees Celsius (float)</li>
            <li><strong>N</strong>: Nitrogen content in soil (float)</li>
            <li><strong>P</strong>: Phosphorus content in soil (float)</li>
            <li><strong>K</strong>: Potassium content in soil (float)</li>
            <li><strong>yeild</strong>: Crop yield (target variable, float)</li>
        </ul>
    </li>
    <li><strong>Missing Values</strong>: None</li>
</ul>
<a>You can Download Dataset from below link also</a>
<href>https://www.kaggle.com/datasets/gurudathg/crop-yield-prediction-using-soil-and-weather</href>

<h3>Data Summary</h3>
<p>The dataset provides a balanced view of critical factors influencing crop yield, such as soil nutrient content, environmental temperature, and fertilizer use. It can be used to train predictive models that guide agricultural decision-making.</p>

<h4>Key Statistics</h4>
<ul>
    <li><strong>Fertilizer</strong>: Range (49.75 to 80.22), Mean: 66.49</li>
    <li><strong>Temperature</strong>: Range (23.77 to 40.27), Mean: 33.85</li>
    <li><strong>Nitrogen (N)</strong>: Range (58.84 to 80.22), Mean: 69.52</li>
    <li><strong>Phosphorus (P)</strong>: Range (17.72 to 25.16), Mean: 20.71</li>
    <li><strong>Potassium (K)</strong>: Range (14.70 to 22.06), Mean: 17.81</li>
    <li><strong>Yield</strong>: Range (5.15 to 12.34), Mean: 8.53</li>
</ul>

<h2>Model Development</h2>
<p>The machine learning model for crop yield prediction is built using regression techniques and leverages the dataset's features. The following steps are taken for model development:</p>
<ol>
    <li><strong>Data Preprocessing</strong>:
        <ul>
            <li>Ensure data normalization and scaling for better model performance.</li>
            <li>Split the dataset into training and testing sets.</li>
        </ul>
    </li>
    <li><strong>Model Selection</strong>:
        <ul>
            <li>Use models such as Linear Regression, Decision Tree, or advanced techniques like Random Forest and Gradient Boosting.</li>
        </ul>
    </li>
    <li><strong>Evaluation Metrics</strong>:
        <ul>
            <li>Assess model performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.</li>
        </ul>
    </li>
    <li><strong>Model Tuning</strong>:
        <ul>
            <li>Apply hyperparameter optimization techniques to fine-tune model performance.</li>
        </ul>
    </li>
</ol>

<h2>How to Use</h2>
<ol>
    <li><strong>Clone the Repository</strong>:
        <pre><code>git clone https://github.com/Gurudath-G/Crop-Prediction-using-ML.git</code></pre>
    </li>
    <li><strong>Load the Dataset</strong>:
        <pre><code>import pandas as pd
df = pd.read_csv('Crop_Yield_Soil_Weather.csv')</code></pre>
    </li>
    <li><strong>Train the Model</strong>:
        <p>Follow the provided Jupyter notebooks or Python scripts to train and evaluate your model using the dataset.</p>
    </li>
</ol>

<h2>Applications</h2>
<ul>
    <li>Predict crop yield based on weather and soil conditions.</li>
    <li>Assist farmers in optimizing fertilizer use for better yield.</li>
    <li>Aid agricultural researchers in developing data-driven growth strategies.</li>
</ul>
