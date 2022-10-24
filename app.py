#crear perceptron con streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import plot_roc_curve

st.title("Perceptron")
st.sidebar.title("Perceptron")
st.markdown("¿Cómo funciona el perceptron?")
st.sidebar.markdown("¿Cómo funciona el perceptron?")
st.markdown("El perceptron es un algoritmo de aprendizaje supervisado que se utiliza para clasificar o separar dos o más clases de datos. El perceptron es un modelo de clasificación lineal, es decir, que se basa en la separación de clases mediante una línea recta. El perceptron es un modelo de clasificación binaria, es decir, que solo puede clasificar dos clases de datos. El perceptron es un modelo de clasificación lineal, es decir, que se basa en la separación de clases mediante una línea recta. El perceptron es un modelo de clasificación binaria, es decir, que solo puede clasificar dos clases de datos.")


#crear perceptron interactivo 
st.sidebar.subheader("Configuración del perceptron")
def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Matriz de confusión")
        plot_confusion_matrix(model, X_test, y_test, display_labels=class_names)
        st.pyplot()
    if 'ROC Curve' in metrics_list:
        st.subheader("Curva ROC")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()
    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Curva Precision-Recall")
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()

def main():
    st.sidebar.subheader("Configuración del perceptron")
    data = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
    split_size = st.sidebar.slider("Tamaño de la muestra", 10, 90, 80, 5)
    metrics = st.sidebar.multiselect("¿Qué métricas deseas ver?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    if data is not None:
        df = pd.read_csv(data)
        class_names = ['0', '1']
        st.dataframe(df)
        if st.button("Clasificar"):
            st.subheader("Datos de entrenamiento")
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size/100)
            st.write("Tamaño de la muestra de entrenamiento:", X_train.shape)
            st.write("Tamaño de la muestra de prueba:", X_test.shape)
            st.subheader("Datos de prueba")
            st.write(X_test)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = Perceptron()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Precisión:", accuracy.round(2))
            st.write("Matriz de confusión:", confusion_matrix(y_test, y_pred))
            st.write("Reporte de clasificación:", classification_report(y_test, y_pred))
            plot_metrics(metrics)
if __name__ == '__main__':
    main()

# Path: requirements.txt

