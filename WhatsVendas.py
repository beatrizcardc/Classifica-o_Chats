import streamlit as st
import pandas as pd
from transformers import pipeline

# Tente carregar um modelo já ajustado para classificação
@st.cache_resource
def carregar_classificador_pt():
    try:
        return pipeline("text-classification", model="pierreguillou/bert-base-cased-sentiment")
    except Exception as e:
        st.error("Erro ao carregar o modelo ajustado. Usando fallback manual.")
        return None

classificador = carregar_classificador_pt()

# Fallback: Regras simples
def categorizar_mensagem_fallback(mensagem):
    texto = mensagem.lower()
    if "cadastro" in texto:
        return "Erro_Dados_Cadastro"
    elif "pagamento" in texto:
        return "Erro_Pagamento"
    else:
        return "Outros"

# Processar mensagens com fallback
def categorizar_mensagem(mensagem):
    if classificador:
        try:
            resultado = classificador(mensagem)
            return resultado[0]['label']
        except Exception:
            pass
    return categorizar_mensagem_fallback(mensagem)

# Função para processar arquivo de conversa
def processar_conversas(conteudo_txt):
    linhas = conteudo_txt.decode("utf-8").split('\n')
    dados = []
    for linha in linhas:
        if ' - ' in linha:
            parte_data, parte_msg = linha.split(' - ', 1)
            if ': ' in parte_msg:
                emissor, mensagem = parte_msg.split(': ', 1)
                dados.append([parte_data, emissor, mensagem])

    df = pd.DataFrame(dados, columns=['Data', 'Emissor', 'Mensagem'])
    df['Categoria'] = df['Mensagem'].apply(categorizar_mensagem)
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


