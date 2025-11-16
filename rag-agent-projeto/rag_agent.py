import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import bs4

# Configurar User Agent
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

load_dotenv()

print("🚀 INICIANDO AGENTE RAG COM LANGCHAIN\n")
print("="*80)

# ========================================
# PASSO 1: INICIALIZAR MODELOS
# ========================================
print("\nInicializando modelos...")
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.7)
embeddings = MistralAIEmbeddings(model="mistral-embed")
print("Modelos inicializados!")

# ========================================
# PASSO 2: CARREGAR DOCUMENTOS
# ========================================
print("\nCarregando documentos...")
url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(web_paths=(url,), bs_kwargs={"parse_only": bs4_strainer})
docs = loader.load()
print(f"{len(docs)} documento(s) carregado(s)")
print(f"Total de caracteres: {len(docs[0].page_content)}")

# ========================================
# PASSO 3: DIVIDIR DOCUMENTOS
# ========================================
print("\nDividindo documentos em chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
splits = text_splitter.split_documents(docs)
print(f"Documento dividido em {len(splits)} chunks")

# ========================================
# PASSO 4: CRIAR E POPULAR VECTOR STORE
# ========================================
print("\nCriando Vector Store e armazenando vetores...")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(splits)
print(f"{len(splits)} vetores armazenados!")

# ========================================
# PASSO 5: CRIAR RETRIEVER
# ========================================
print("\nCriando retriever...")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("Retriever criado!")

# ========================================
# PASSO 6: CRIAR PROMPT TEMPLATE
# ========================================
print("\nCriando template de prompt...")
template = """Você é um assistente especializado que responde perguntas baseado no contexto fornecido.

Contexto:
{context}

Pergunta: {question}

Responda de forma clara e detalhada em português, usando apenas as informações do contexto acima.
Se a informação não estiver no contexto, diga que não sabe.

Resposta:"""

prompt = ChatPromptTemplate.from_template(template)
print("Template criado!")

# ========================================
# PASSO 7: CRIAR CHAIN RAG
# ========================================
print("\nCriando chain RAG...")

def format_docs(docs):
    """Formata documentos recuperados"""
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("Chain RAG criada!")

# ========================================
# PASSO 8: FAZER PERGUNTAS
# ========================================
print("\n" + "="*80)
print("INICIANDO CONSULTAS")
print("="*80)

perguntas = [
    "Qual é o método padrão para decomposição de tarefas?",
    "Quais são as extensões comuns do método Chain of Thought?",
    "O que é Tree of Thoughts e como funciona?"
]

for i, pergunta in enumerate(perguntas, 1):
    print(f"\n{'='*80}")
    print(f"PERGUNTA {i}: {pergunta}")
    print(f"{'='*80}\n")
    
    try:
        # Recuperar documentos relevantes
        docs_relevantes = retriever.invoke(pergunta)
        print(f"Recuperados {len(docs_relevantes)} documentos relevantes")
        
        # Gerar resposta
        print("\nGerando resposta...\n")
        resposta = rag_chain.invoke(pergunta)
        
        print(f"RESPOSTA:")
        print(f"{resposta}")
        
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "-"*80)

print("\nCONSULTAS CONCLUÍDAS!")
print("="*80)

# ========================================
# MODO INTERATIVO (OPCIONAL)
# ========================================
print("\n" + "="*80)
print("MODO INTERATIVO")
print("="*80)
print("Digite suas perguntas (ou 'sair' para encerrar)\n")

while True:
    pergunta = input("Você: ").strip()
    
    if pergunta.lower() in ['sair', 'exit', 'quit', '']:
        print("\nEncerrando...")
        break
    
    try:
        print("\nProcessando...\n")
        resposta = rag_chain.invoke(pergunta)
        print(f"Assistente: {resposta}\n")
        print("-"*80 + "\n")
    except Exception as e:
        print(f"Erro: {e}\n")
