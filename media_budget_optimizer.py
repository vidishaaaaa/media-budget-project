import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# Sample dataset creation
sample_data = pd.DataFrame({
    'Week': list(range(1, 21)),
    'TV_Spend': [35, 45, 28, 40, 33, 39, 48, 44, 31, 36, 30, 42, 37, 41, 46, 38, 43, 29, 34, 32],
    'Facebook_Spend': [15, 20, 12, 18, 14, 19, 24, 21, 16, 17, 13, 23, 22, 25, 20, 14, 18, 12, 15, 16],
    'Google_Spend': [50, 55, 45, 52, 48, 53, 60, 58, 47, 51, 46, 59, 54, 57, 56, 49, 52, 44, 50, 48]
})

# Generate Sales based on a linear combination of spends plus noise
sample_data['Sales'] = (
    2 * sample_data['TV_Spend'] +
    3 * sample_data['Facebook_Spend'] +
    4 * sample_data['Google_Spend'] +
    np.random.normal(0, 10, size=20)
)

# Load dataset
data = sample_data

# Model training
features = ['TV_Spend', 'Facebook_Spend', 'Google_Spend']
target = 'Sales'

X = data[features]
y = data[target]

model = LinearRegression()
model.fit(X, y)

# Streamlit App
st.title('Media Budget Optimizer & Campaign Performance Dashboard')

st.subheader('Upload Your Campaign Data (or use sample below)')
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    X = data[features]
    y = data[target]
    model.fit(X, y)

st.write('### Campaign Data', data)

# Display model coefficients
st.subheader('Channel Effectiveness (ROI per unit spend)')
coefficients = pd.DataFrame({'Channel': features, 'Effectiveness': model.coef_})
st.write(coefficients)

# Budget allocation inputs
st.subheader('Adjust Budget Allocation')
tv_budget = st.slider('TV Budget', 0, 100, int(data['TV_Spend'].mean()))
facebook_budget = st.slider('Facebook Budget', 0, 100, int(data['Facebook_Spend'].mean()))
google_budget = st.slider('Google Budget', 0, 100, int(data['Google_Spend'].mean()))

# Prediction based on new budget
new_spend = np.array([[tv_budget, facebook_budget, google_budget]])
predicted_sales = model.predict(new_spend)[0]

st.write(f'### Predicted Sales: {predicted_sales:.2f} units')

# Visualization
st.subheader('Current vs Adjusted Spend')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
budget_labels = ['TV', 'Facebook', 'Google']
current_spend = [data['TV_Spend'].mean(), data['Facebook_Spend'].mean(), data['Google_Spend'].mean()]
adjusted_spend = [tv_budget, facebook_budget, google_budget]

x = np.arange(len(budget_labels))
width = 0.35

rects1 = ax.bar(x - width/2, current_spend, width, label='Current')
rects2 = ax.bar(x + width/2, adjusted_spend, width, label='Adjusted')

ax.set_ylabel('Budget Spend')
ax.set_title('Current vs Adjusted Spend')
ax.set_xticks(x)
ax.set_xticklabels(budget_labels)
ax.legend()

st.pyplot(fig)

st.write('---')
st.write('This dashboard helps you reallocate your marketing budget to maximize predicted sales based on historical performance.')
