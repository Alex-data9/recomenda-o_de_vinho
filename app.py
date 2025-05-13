import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# 🔥 Colunas EXATAS que existem no CSV (com underscores)
feature_cols = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol"
]

@st.cache_data
def load():
    # 1) Carrega o DataFrame completo
    df = pd.read_csv("wine_quality_full.csv")
    
    # 2) Carrega o scaler que você treinou anteriormente
    scaler = joblib.load("scaler.joblib")
    
    # 3) Extrai apenas as colunas corretas
    features = df[feature_cols]
    
    # 4) Ajusta o NearestNeighbors só nessas features
    nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    nn.fit(scaler.transform(features))
    
    return df, scaler, nn

# Carrega tudo de uma vez
df, scaler, nn = load()

st.title("🍷 Recomendador de Vinhos")

# Input de índice
idx = st.number_input(
    "Índice do vinho de referência",
    min_value=0, max_value=len(df)-1
)

if st.button("Recomendar"):
    # De novo, seleciona as MESMAS colunas
    X = df[feature_cols]
    X_scaled = scaler.transform(X)

    # Busca os 6 mais próximos
    dists, inds = nn.kneighbors([X_scaled[idx]])
    recs = df.iloc[inds[0]].drop(idx)

    st.write("#### Sua escolha:")
    st.write(df.iloc[idx])
    st.write("#### Recomendações:")
    st.dataframe(recs)

    faixa_alco = st.slider("Teor alcoólico", min_value=8.0, max_value=14.0, step=0.1)
subset = df[df["alcohol"] <= faixa_alco]
st.dataframe(subset)
