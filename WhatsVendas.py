import streamlit as st
import pandas as pd
from transformers import pipeline
from datetime import datetime

# Tente carregar um modelo já ajustado para classificação
@st.cache_resource
def carregar_classificador_pt():
    try:
        return pipeline("text-classification", model="neuralmind/bert-base-portuguese-cased")
    except Exception as e:
        st.error("Erro ao carregar o modelo ajustado. Certifique-se de que o modelo está acessível.")
        return None

classificador = carregar_classificador_pt()

# Fallback: Regras simples
def categorizar_mensagem_fallback(mensagem):
    texto = mensagem.lower()
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
    elif "liberar" in texto and "plano de vendas" in texto:
        return "Pedido_LiberarPlanoVenda"
    else:
        return "Outros"

# Categorização baseada na escolha do método
def categorizar_mensagem(mensagem, metodo):
    if metodo == "modelo IA":
        if classificador:
            try:
                resultado = classificador(mensagem)
                return resultado[0]['label']
            except Exception:
                st.warning("Erro ao usar o modelo de IA. Retornando 'Outros'.")
                return "Outros"
        else:
            st.warning("Modelo de IA não carregado. Usando 'Outros'.")
            return "Outros"
    else:  # Fallback por regra de mensagem
        return categorizar_mensagem_fallback(mensagem)

# Função para processar arquivo de conversa
def processar_conversas(conteudo_txt, metodo):
    linhas = conteudo_txt.decode("utf-8").split('\n')
    dados = []
    for linha in linhas:
        if ' - ' in linha:
            parte_data, parte_msg = linha.split(' - ', 1)
            if ': ' in parte_msg:
                emissor, mensagem = parte_msg.split(': ', 1)
                try:
                    # Converter data para formato datetime
                    data = datetime.strptime(parte_data, "%d/%m/%Y %H:%M")
                    dados.append([data, emissor, mensagem])
                except ValueError:
                    continue

    df = pd.DataFrame(dados, columns=['Data', 'Emissor', 'Mensagem'])

    # Classificar mensagens com o método escolhido
    df['Categoria'] = df['Mensagem'].apply(lambda x: categorizar_mensagem(x, metodo))
    return df

# Interface do Streamlit
st.title("Classificação de Conversas do WhatsApp")
st.sidebar.header("Opções")

# Escolha do método de classificação
metodo_classificacao = st.sidebar.radio(
    "Escolha o método de classificação:",
    ("modelo IA", "Regra de Mensagem")
)

# Upload do arquivo
uploaded_file = st.file_uploader("Faça upload do arquivo de conversa (.txt)", type="txt")
if uploaded_file is not None:
    st.write("Processando arquivo...")
    df_conversas = processar_conversas(uploaded_file.read(), metodo_classificacao)

    # Seleção de período de data
    min_data = df_conversas['Data'].min()
    max_data = df_conversas['Data'].max()

    st.sidebar.write("Selecione o período de data:")
    data_inicial = st.sidebar.date_input("Data Inicial", min_data.date())
    data_final = st.sidebar.date_input("Data Final", max_data.date())

    if data_inicial > data_final:
        st.sidebar.error("A data inicial não pode ser maior que a data final.")
    else:
        df_conversas = df_conversas[(df_conversas['Data'] >= pd.Timestamp(data_inicial)) &
                                    (df_conversas['Data'] <= pd.Timestamp(data_final))]

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


    st.write("Número de Incidentes por Categoria")
    grafico_categorias = df_conversas['Categoria'].value_counts()
    st.bar_chart(grafico_categorias)


