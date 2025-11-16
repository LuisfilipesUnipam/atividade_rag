## 📋 Descrição do Projeto

Este projeto implementa um agente conversacional que:
- Carrega e processa documentos de uma fonte web
- Busca informações relevantes usando busca vetorial
- Gera respostas contextualizadas usando LLM
- Permite interação em tempo real com o usuário

### Tema Escolhido
**Agentes Autônomos baseados em LLMs**, incluindo conceitos como:
- Task Decomposition (Decomposição de Tarefas)
- Chain of Thought (CoT)
- Tree of Thoughts (ToT)
- Planejamento e Reflexão em Agentes

### Fonte de Dados
Blog post: [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) por Lilian Weng

---

## Tecnologias Utilizadas

| Componente | Tecnologia |
|------------|------------|
| **Framework** | LangChain |
| **Provedor LLM** | Mistral AI |
| **Modelo de Chat** | mistral-small-latest |
| **Modelo de Embeddings** | mistral-embed |
| **Document Loader** | WebBaseLoader (BeautifulSoup4) |
| **Text Splitter** | RecursiveCharacterTextSplitter |
| **Vector Store** | InMemoryVectorStore |
| **Linguagem** | Python 3.12+ |

---

## Instalação

### 1 Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/rag-agent-projeto.git
cd rag-agent-projeto```

```bash
git clone https://github.com/seu-usuario/rag-agent-projeto.git
cd rag-agent-projeto
```

### 2 Criar Ambiente Virtual

# Criar ambiente virtual
```bash
python -m venv venv
```

# Ativar ambiente virtual
# Linux/Mac:
```
source venv/bin/activate
```

# Windows:
```
venv\Scripts\activate
```

### 3️3 Instalar Dependências

```bash
Copy
pip install -r requirements.txt
```

## Configuração

### 1 Obter API Key do Mistral AI

Acesse: https://console.mistral.ai/
Crie uma conta (pode usar GitHub)
Vá em "API Keys" no menu lateral
Clique em "Create new key"
Copie a chave gerada

### 3 Configurar Arquivo .env

Crie um arquivo .env na raiz do projeto:
``` 
bash
Copy
MISTRAL_API_KEY=sua_chave_mistral_aqui
```
⚠️ IMPORTANTE: Nunca compartilhe sua API key publicamente!

### 🚀 Como Executar
Executar o Agente
```bash
Copy
python rag_agent.py
```