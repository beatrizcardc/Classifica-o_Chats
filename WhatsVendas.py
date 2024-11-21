import streamlit as st
import pandas as pd
from transformers import pipeline

# Configuração do modelo BERT
@st.cache_resource
def carregar_classificador():
    return pipeline("text-classification", model="bert-base-uncased")

classificador = carregar_classificador()

# Função para processar o arquivo de conversa
def processar_conversas(conteudo_txt):
    linhas = conteudo_txt.decode("utf-8").split('\n')

    dados = []
    for linha in linhas:
        if ' - ' in linha:  # Exemplo: "20/11/2024 11:55 - João: Mensagem"
            parte_data, parte_msg = linha.split(' - ', 1)
            if ': ' in parte_msg:
                emissor, mensagem = parte_msg.split(': ', 1)
                dados.append([parte_data, emissor, mensagem])

    df = pd.DataFrame(dados, columns=['Data', 'Emissor', 'Mensagem'])

    # Classificar mensagens usando o modelo BERT
    def classificar_mensagem(mensagem):
        try:
            resultado = classificador(mensagem)
            return resultado[0]['label']
        except Exception:
            return "Erro"

    df['Categoria'] = df['Mensagem'].apply(classificar_mensagem)
    return df

# Interface do Streamlit
st.title("Classificação de Conversas do WhatsApp")
st.sidebar.header("Opções")

# Upload do arquivo
uploaded_file = st.file_uploader("Faça upload do arquivo de conversa (.txt)", type="txt")
if uploaded_file is not None:
    st.write("Processando arquivo...")
    df_conversas = processar_conversas(uploaded_file.read())

    # Filtros no menu lateral
    categorias = st.sidebar.multiselect("Selecione as categorias", df_conversas['Categoria'].unique())
    emissores = st.sidebar.multiselect("Selecione os emissores", df_conversas['Emissor'].unique())

    # Aplicar filtros
    if categorias:
        df_conversas = df_conversas[df_conversas['Categoria'].isin(categorias)]
    if emissores:
        df_conversas = df_conversas[df_conversas['Emissor'].isin(emissores)]

    # Exibir tabela e gráficos
    st.write("Conversas Categorizadas")
    st.dataframe(df_conversas)

    st.write("Número de Incidentes por Categoria")
    grafico_categorias = df_conversas['Categoria'].value_counts()
    st.bar_chart(grafico_categorias)
