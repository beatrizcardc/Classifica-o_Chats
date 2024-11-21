import streamlit as st
import pandas as pd
from transformers import pipeline

# Carregar o classificador em Português
@st.cache_resource
def carregar_classificador_pt():
    return pipeline("text-classification", model="neuralmind/bert-base-portuguese-cased")

classificador = carregar_classificador_pt()

# Categorias Personalizadas
categorias_personalizadas = [
    "Erro_Endereço_Cadastro", "Erro_Nome_Cadastro", "Erro_Documento_Cadastro",
    "Erro_Dados_Cadastro", "Erro_Pagamento_Boleto", "Erro_Pagamento_Crédito",
    "Erro_Pagamento_Link", "Erro_Envio_Contrato", "Erro_Assinatura_Contrato",
    "Erro_Acesso_NewConWeb", "Erro_Acesso_Plataforma", "Lentidão_Plataforma",
    "Lentidão_NewCon", "Pedido_AumentoLimiteCreditoPV",
    "Pedido_AumentoLimiteCreditoPessoa", "Pedido_AlteraçãoEmail",
    "Pedido_AlteraçãoDados"
]

# Função para atribuir categorias
def categorizar_mensagem(mensagem):
    # Análise de texto com o modelo
    resultado = classificador(mensagem)
    texto = mensagem.lower()
    
    # Regras simples para identificar categorias
    if "endereço" in texto and "cadastro" in texto:
        return "Erro_Endereço_Cadastro"
    elif "nome" in texto and "cadastro" in texto:
        return "Erro_Nome_Cadastro"
    elif "documento" in texto and "cadastro" in texto:
        return "Erro_Documento_Cadastro"
    elif "dados" in texto and "cadastro" in texto:
        return "Erro_Dados_Cadastro"
    elif "boleto" in texto and "pagamento" in texto:
        return "Erro_Pagamento_Boleto"
    elif "crédito" in texto and "pagamento" in texto:
        return "Erro_Pagamento_Crédito"
    elif "link" in texto and "pagamento" in texto:
        return "Erro_Pagamento_Link"
    elif "envio" in texto and "contrato" in texto:
        return "Erro_Envio_Contrato"
    elif "assinatura" in texto and "contrato" in texto:
        return "Erro_Assinatura_Contrato"
    elif "acesso" in texto and "newconweb" in texto:
        return "Erro_Acesso_NewConWeb"
    elif "acesso" in texto and "plataforma" in texto:
        return "Erro_Acesso_Plataforma"
    elif "lentidão" in texto and "plataforma" in texto:
        return "Lentidão_Plataforma"
    elif "lentidão" in texto and "newcon" in texto:
        return "Lentidão_NewCon"
    elif "aumento" in texto and "limite" in texto and "pv" in texto:
        return "Pedido_AumentoLimiteCreditoPV"
    elif "aumento" in texto and "limite" in texto and "pessoa" in texto:
        return "Pedido_AumentoLimiteCreditoPessoa"
    elif "alteração" in texto and "email" in texto:
        return "Pedido_AlteraçãoEmail"
    elif "alteração" in texto and "dados" in texto:
        return "Pedido_AlteraçãoDados"
    else:
        return "Outros"

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

    # Classificar mensagens usando regras e modelo
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

