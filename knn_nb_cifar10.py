
import streamlit as st                       # webpage
import numpy as np                           # array handeling
import random                                # random number
import matplotlib.pyplot as plt               #Graphs
from tensorflow.keras.datasets import cifar10 #load image from cifar10
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix    #performance validation
import seaborn as sns                           # for heatmap
import pandas as pd                            # for table

# UI backgrounnd^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
plt.style.use('dark_background')
colors_palette = ['#9D4EDD', '#7B2CBF', '#00D084', '#2CE28D', '#5A189A']

# Heading ****************************************************************************
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

st.title("CIFAR-10 Classification using k-NN and Naive Bayes")

# **************************************************************************************************
# Load Data
@st.cache_data
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train[:5000], y_train[:5000], X_test[:1000], y_test[:1000]

X_train, y_train, X_test, y_test = load_data()

class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# -----------------------------
# Train Models
# -----------------------------
@st.cache_resource
def train_models(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    nb = GaussianNB()
    nb.fit(X_train, y_train)

    return knn, nb

knn, nb = train_models(X_train, y_train)

# -----------------------------
# Accuracy Section
# -----------------------------
knn_accuracy = accuracy_score(y_test, knn.predict(X_test))
nb_accuracy = accuracy_score(y_test, nb.predict(X_test))

st.subheader("Model Accuracy")
# Get predictions
knn_pred = knn.predict(X_test)
nb_pred = nb.predict(X_test)

# matrix display++++++++++++++++++++++++++++++++++++++++++++++++++++++
col1, col2 = st.columns(2)
col1.metric("k-NN Accuracy", f"{knn_accuracy:.4f}")
col2.metric("Naive Bayes Accuracy", f"{nb_accuracy:.4f}")

# -----------------------------
# Detailed Analysis (Confusion Matrix & Class Accuracy)
# -----------------------------
st.divider()
st.header("Detailed Analysis")

# Function to calculate per-class accuracy
def get_class_accuracy(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    return dict(zip(classes, class_acc))

knn_class_acc = get_class_accuracy(y_test, knn_pred, class_names)
nb_class_acc = get_class_accuracy(y_test, nb_pred, class_names)

# --- Accuracy per Class logic moved to end ---

# Confusion Matrices
st.subheader("Confusion Matrices")
tab1, tab2 = st.tabs(["k-NN", "Naive Bayes"])

with tab1:
    fig_cm_knn, ax_cm_knn = plt.subplots(figsize=(8, 6), facecolor='#0e1117')
    cm_knn = confusion_matrix(y_test, knn_pred)
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Purples', 
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm_knn)
    ax_cm_knn.set_title("k-NN Confusion Matrix", color='white')
    ax_cm_knn.set_xlabel("Predicted", color='white')
    ax_cm_knn.set_ylabel("True", color='white')
    st.pyplot(fig_cm_knn)

with tab2:
    fig_cm_nb, ax_cm_nb = plt.subplots(figsize=(8, 6), facecolor='#0e1117')
    cm_nb = confusion_matrix(y_test, nb_pred)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=ax_cm_nb)
    ax_cm_nb.set_title("Naive Bayes Confusion Matrix", color='white')
    ax_cm_nb.set_xlabel("Predicted", color='white')
    ax_cm_nb.set_ylabel("True", color='white')
    st.pyplot(fig_cm_nb)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# (Already moved up in the previous chunk for logic flow)
# knn_pred = knn.predict(X_test)
# nb_pred = nb.predict(X_test)

# Graph 1: Model Comparison (Colorful Bar Chart)@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
st.subheader("Model Performance Comparison")
fig1, ax1 = plt.subplots(figsize=(5, 2.5), facecolor='#0e1117', edgecolor='#9D4EDD')
bars = ax1.bar(["k-NN", "Naive Bayes"], [knn_accuracy, nb_accuracy], 
               color=['#9D4EDD', '#00D084'], edgecolor='white', linewidth=2)
ax1.set_ylim(0, 1)
ax1.set_ylabel("Accuracy", fontsize=10, color='white', weight='bold')
ax1.set_title("Overall Model Accuracy", fontsize=12, color='white', weight='bold')
ax1.tick_params(axis='both', labelsize=9, colors='white')
ax1.set_facecolor('#1e1e2e')
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10, color='white', weight='bold')
st.pyplot(fig1, use_container_width=False)
#_@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Graph 2: Error Rate Comparison (Pie Charts)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
st.subheader("Error Rate Distribution")
col1, col2 = st.columns(2)

with col1:
    fig4, ax4 = plt.subplots(figsize=(4, 3), facecolor='#0e1117')
    knn_error = 1 - knn_accuracy
    sizes = [knn_accuracy, knn_error]
    colors_pie = ['#00D084', '#9D4EDD']
    wedges, texts, autotexts = ax4.pie(sizes, labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90, textprops={'color': 'white', 'weight': 'bold'})
    ax4.set_title("k-NN Error Rate", fontsize=11, color='white', weight='bold')
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    st.pyplot(fig4, use_container_width=False)

with col2:
    fig5, ax5 = plt.subplots(figsize=(4, 3), facecolor='#0e1117')
    nb_error = 1 - nb_accuracy
    sizes = [nb_accuracy, nb_error]
    colors_pie = ['#2CE28D', '#7B2CBF']
    wedges, texts, autotexts = ax5.pie(sizes, labels=['Correct', 'Incorrect'], autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90, textprops={'color': 'white', 'weight': 'bold'})
    ax5.set_title("Naive Bayes Error Rate", fontsize=11, color='white', weight='bold')
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(10)
        autotext.set_weight('bold')
    st.pyplot(fig5, use_container_width=False)

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Random Image Prediction
# -----------------------------
st.subheader("Random Image Prediction")

if st.button("Generate Random Prediction"):

    index = random.randint(0, len(X_test) - 1)

    image = X_test[index].reshape(32, 32, 3)
    true_label = class_names[y_test[index]]

    knn_result = class_names[knn.predict(X_test[index].reshape(1, -1))[0]]
    nb_result = class_names[nb.predict(X_test[index].reshape(1, -1))[0]]

    st.image(image, caption="Test Image", width=220)

    st.write("True Label:", true_label)
    st.write("k-NN Prediction:", knn_result)
    st.write("Naive Bayes Prediction:", nb_result)

# -----------------------------
# Final Per-Class Accuracy Analysis
# -----------------------------
st.divider()
st.subheader("Final Performance Breakdown: Accuracy per Class")

# Create a styled dataframe for per-class accuracy
acc_df = pd.DataFrame({
    "Class": class_names,
    "k-NN Accuracy": [f"{knn_class_acc[c]*100:.1f}%" for c in class_names],
    "Naive Bayes Accuracy": [f"{nb_class_acc[c]*100:.1f}%" for c in class_names]
})

# Display as a clean table
st.table(acc_df)

# Optional: Add a small analysis note
best_knn = max(knn_class_acc, key=knn_class_acc.get)
best_nb = max(nb_class_acc, key=nb_class_acc.get)
st.success("Execution completed successfully")
