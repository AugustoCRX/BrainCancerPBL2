import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
import pandas as pd
import os
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

path = os.getcwd()

model_dict = {
    'Inception': ('inception_model.keras', 'inception_history.pkl'),
    'ResNet': ('resnet_model.keras', 'resnet_history.pkl')
}

classes_dict = {
    0: 'Glioma',
    1: 'Meningioma',
    2: 'Tumor benigno'
}

def load_files(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_models_and_histories():
    models = {}
    model_history = {}
    
    for model_name, (model_file, history_file) in model_dict.items():
        models[model_name] = tf.keras.models.load_model(
            f'{path}/models/{model_file}', 
            compile=False
        )
        
        history_path = f'{path}/models/{history_file}'
        model_history[model_name] = load_files(history_path)

    return models, model_history

selected_model = st.selectbox(
    "Selecione o modelo:",
    options=list(model_dict.keys()),
    key="model_selector"
)

models, model_history = load_models_and_histories()

model = models[selected_model]
history = model_history[selected_model]
pred_dict = load_files(f'{path}/data/processed/pred_dict.pkl')
y_test = pred_dict['y_true']
y_pred_inception = np.argmax(pred_dict['y_pred_imodel'], axis=1)
y_pred_resnet = np.argmax(pred_dict['y_pred_rmodel'], axis=1)

uploaded_file = st.file_uploader("Carregue uma imagem...", type=['jpg', 'png'])

if uploaded_file is not None:

    img = Image.open(uploaded_file).resize((224, 224))
    img_array = np.expand_dims(img, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    class_name = classes_dict[class_idx]
    confidence = prediction[0][class_idx]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img, caption='Imagem carregada', width=200)
        st.write(f"**Classe predita:** {class_name}")
        st.write(f"**Confiança:** {confidence:.2%}")

metric_map = {
    "Perda": ("loss", "val_loss", "Erro durante o treinamento"),
    "Acurácia": ("accuracy", "val_accuracy", "Desempenho do modelo"),
    "Acurácia Categórica": ("categorical_accuracy", "val_categorical_accuracy", "Acurácia Categórica")
}

col1, col2 = st.columns(2)
with col1:
    show_train = st.checkbox("Mostrar dados de Treino", value=True)
with col2:
    show_val = st.checkbox("Mostrar dados de Validação", value=True)

metric = st.selectbox(
    "Selecione a métrica:",
    options=["Perda", "Acurácia", "Acurácia Categórica", "Matriz de Confusão"],
    key="metric_selector"
)

if metric == "Matriz de Confusão":
    pass
else:
    compare_model = st.checkbox("➕ Comparar com outro modelo", value = False)
if metric:
    if metric == "Matriz de Confusão":
        if selected_model == 'Inception':
            y_pred = y_pred_inception
        else:
            y_pred = y_pred_resnet
        
        if len(y_pred.shape) == 2:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            y_pred_classes = y_pred

        cm = confusion_matrix(y_test, y_pred_classes)
        
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        
        ax.set(
            xticks=np.arange(3),
            yticks=np.arange(3),
            xticklabels=[i for i in classes_dict.values()],
            yticklabels=[i for i in classes_dict.values()],
            title=f'Matriz de Confusão - {selected_model}',
            ylabel='Rótulo Verdadeiro',
            xlabel='Rótulo Predito',
        )
        
        thresh = cm.max() / 2.
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        st.pyplot(fig)
    
    else:
        train_col, val_col, title = metric_map[metric]
        
        fig, ax = plt.subplots(figsize=(10, 9))
        colors = ['#1f77b4', '#6b6b6b']
        
        models_to_plot = [selected_model]
        if compare_model:
            other_model = [m for m in model_dict.keys() if m != selected_model][0]
            models_to_plot.append(other_model)
        
        for idx, model_name in enumerate(models_to_plot):
            hist = pd.DataFrame(model_history[model_name].history)
            
            if show_train:
                ax.plot(
                    hist[train_col], 
                    color=colors[idx],
                    linestyle='-',
                    markersize=6,
                    linewidth=2.5,
                    label=f'{model_name} - Treino'
                )
            
            if show_val and val_col in hist.columns:
                ax.plot(
                    hist[val_col], 
                    color=colors[idx],
                    linestyle='--',
                    markersize=6,
                    linewidth=2.5,
                    alpha=0.9,
                    label=f'{model_name} - Validação'
                )
        
        ax.set_title(f"{title} - Comparação entre Modelos", fontsize=14, pad=20)
        ax.set_xlabel("Época", fontsize=12, labelpad=10)
        ax.set_ylabel(metric, fontsize=12, labelpad=10)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if show_train or show_val:
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=False,
                fontsize=12
            )
        
        st.pyplot(fig)