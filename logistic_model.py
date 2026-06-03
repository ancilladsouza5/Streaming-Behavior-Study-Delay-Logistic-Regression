import pandas as pd

# Load dataset
data = pd.read_csv("streaming_data.csv")

# Display first 5 rows
print(data.head())
# Define dependent variable (Y)
Y = data["Delay-Code"]

# Define independent variables (X)
X = data[["Age-Code", "Frequency-Code", "Time-Code", "Overwatch-Code"]]

# Display variables
print(X.head())
print(Y.head())

import statsmodels.api as sm

# Add constant term (intercept)
X = sm.add_constant(X)

# Logistic Regression Model
logit_model = sm.Logit(Y, X)

# Fit model
result = logit_model.fit()

# Display results
print(result.summary())

from sklearn.metrics import confusion_matrix, accuracy_score

# Predicted probabilities
y_pred_prob = result.predict(X)

# Convert probabilities into 0 and 1
y_pred = (y_pred_prob >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(Y, y_pred)

# Accuracy
accuracy = accuracy_score(Y, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nAccuracy Score:")
print(accuracy)

import matplotlib.pyplot as plt

# Variables
variables = ["Age", "Frequency", "Time", "Overwatching"]

# Regression coefficients
coefficients = [-0.0309, 0.3632, 0.3844, 0.4903]

# Create figure
plt.figure(figsize=(8,5))

# Bar chart
plt.bar(variables, coefficients)

# Labels and title
plt.xlabel("Variables")
plt.ylabel("Coefficient Value")
plt.title("Logistic Regression Coefficients")

# Horizontal line at zero
plt.axhline(0)

# Save graph as PNG
plt.savefig("streaming_coefficients.png", dpi=300, bbox_inches='tight')

# Show graph
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Create confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot
disp.plot()

# Title
plt.title("Confusion Matrix")

# Save as PNG
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')

# Show graph
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Regression results table
results_table = pd.DataFrame({
    "Variable": ["Constant", "Age", "Streaming Frequency", "Streaming Time", "Overwatching"],
    "Coefficient": [-2.922, -0.031, 0.363, 0.384, 0.490],
    "Std_Error": [1.167, 0.240, 0.301, 0.195, 0.209],
    "P_Value": [0.012, 0.897, 0.227, 0.049, 0.019]
})

# Create figure
fig, ax = plt.subplots(figsize=(10,3))

# Hide axes
ax.axis('off')

# Create table
table = ax.table(
    cellText=results_table.values,
    colLabels=results_table.columns,
    loc='center'
)

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Title
plt.title("Binary Logistic Regression Results", pad=20)

# Save as PNG
plt.savefig("logistic_regression_table.png",
            dpi=300,
            bbox_inches='tight')

# Show table
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Remove constant for VIF calculation
X_vif = data[["Age-Code", "Frequency-Code", "Time-Code", "Overwatch-Code"]]

# Add constant
X_vif_const = sm.add_constant(X_vif)

# Create VIF dataframe
vif_data = pd.DataFrame()
vif_data["Variable"] = X_vif_const.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_vif_const.values, i)
    for i in range(X_vif_const.shape[1])
]

# Display VIF table
print(vif_data)

from sklearn.metrics import roc_curve, roc_auc_score

# ROC values
fpr, tpr, thresholds = roc_curve(Y, y_pred_prob)

# AUC score
auc_score = roc_auc_score(Y, y_pred_prob)

print("AUC Score:", auc_score)

# Plot ROC Curve
plt.figure(figsize=(7,5))

plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")

# Random line
plt.plot([0,1], [0,1], linestyle='--')

# Labels
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Binary Logistic Regression")

plt.legend()

# Save figure
plt.savefig("roc_curve.png",
            dpi=300,
            bbox_inches='tight')

plt.show()

from sklearn.metrics import classification_report

# Classification report
report = classification_report(Y, y_pred)

print(report)

# Create figure
fig, ax = plt.subplots(figsize=(6,2))

# Hide axes
ax.axis('off')

# Create table
table = ax.table(
    cellText=vif_data.values,
    colLabels=vif_data.columns,
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Title
plt.title("Variance Inflation Factor (VIF)")

# Save image
plt.savefig("vif_table.png",
            dpi=300,
            bbox_inches='tight')

plt.show()