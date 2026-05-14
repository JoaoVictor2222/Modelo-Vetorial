# Documentação Completa — Sistema de Recuperação de Informação (Modelo Vetorial)

## 📋 Índice

1. [Visão Geral do Projeto](#visão-geral-do-projeto)
2. [Conceitos Teóricos](#conceitos-teóricos)
3. [Arquitetura do Sistema](#arquitetura-do-sistema)
4. [Descrição Detalhada dos Scripts](#descrição-detalhada-dos-scripts)
5. [Fluxo de Execução](#fluxo-de-execução)
6. [Exemplos de Uso](#exemplos-de-uso)
7. [Cálculos Implementados](#cálculos-implementados)

---

## Visão Geral do Projeto

### Objetivo

Implementar um **Sistema de Recuperação de Informação (Information Retrieval - IR)** baseado no **Modelo Vetorial** com as seguintes capacidades:

1. **Índice Invertido**: Estrutura eficiente que mapeia cada termo único para os documentos que o contêm
2. **Ponderação TF-IDF**: Atribui pesos aos termos baseado em sua frequência e importância
3. **Similaridade do Cosseno**: Calcula a relevância entre consultas e documentos
4. **Ranking Ordenado**: Ordena resultados por relevância

### Requisitos Atendidos

| Requisito | Nota | Implementação |
|-----------|------|---------------|
| Índice Invertido com frequências | 50% | `index.py` + `cli.py` |
| Representação no Modelo Vetorial | 25% | `vectorizer.py` |
| Cálculo de Similaridade | 15% | `ranker.py` |
| Ranking Ordenado | 10% | `ranker.py` |

---

## Conceitos Teóricos

### 1. Modelo Vetorial (Vector Space Model - VSM)

Cada documento e consulta é representado como um **vetor no espaço multidimensional**, onde cada dimensão representa um termo único da coleção.

```
Documento = [peso_termo1, peso_termo2, ..., peso_termoN]
```

### 2. Índice Invertido

Estrutura de dados que mapeia termos para documentos:

```
Termo "python" → [(doc_0, freq=3), (doc_5, freq=2), (doc_8, freq=1)]
Termo "código" → [(doc_1, freq=4), (doc_3, freq=2)]
```

**Vantagens**:
- Busca rápida de documentos contendo um termo
- Cálculo eficiente de frequência de documentos (DF)

### 3. TF-IDF (Term Frequency - Inverse Document Frequency)

Métrica que reflete a **importância de um termo em um documento** dentro de uma coleção.

**Fórmula implementada**:
```
TF(termo, doc) = 1 + log10(frequência do termo no documento)
IDF(termo) = 1 + log10(N / DF)
      onde N = total de documentos
            DF = número de documentos contendo o termo

TF-IDF = TF × IDF
```

**Intuitivamente**:
- **TF alto**: Termo aparece muitas vezes no documento (relevante para este doc)
- **IDF alto**: Termo é raro na coleção (termo discriminativo)
- **TF-IDF alto**: Termo importante e específico para este documento

### 4. Similaridade do Cosseno

Mede a **similaridade angular entre dois vetores**:

```
Similaridade(V1, V2) = (V1 · V2) / (||V1|| × ||V2||)
```

**Vantagens**:
- Valor entre 0 e 1 (fácil interpretação)
- Independente da magnitude dos vetores
- Eficiente computacionalmente

---

## Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│              (Ponto de Entrada)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                      cli.py                             │
│         (Interface com Usuário)                         │
├─────────────────────────────────────────────────────────┤
│  • select_collection()      - Seleciona pasta           │
│  • load_documents()         - Carrega arquivos .txt     │
│  • build_index()            - Constrói índice invertido │
│  • vectorize_documents()    - Calcula TF-IDF            │
│  • query_loop()             - Loop de consultas         │
└────────────┬────────────────────────────┬──────────────┘
             │                            │
      ┌──────▼──────┐            ┌────────▼────────┐
      │              │            │                 │
      ▼              ▼            ▼                 ▼
┌───────────┐ ┌──────────────┐ ┌────────┐  ┌──────────┐
│preprocess-│ │  document.py │ │index.py│  │vectorizer│
│   or.py   │ │              │ │        │  │   .py    │
│           │ │ class:       │ │class:  │  │ class:   │
│ class:    │ │ Document     │ │Inverted│  │ Vectorizer
│Processor  │ │              │ │Index   │  │          │
│           │ │ Atributos:   │ │        │  │ Funções: │
│ Funções:  │ │ • doc_id     │ │index   │  │ • tf()   │
│ • tokenize│ │ • filename   │ │doc_freq│  │ • idf()  │
│ • remove_ │ │ • text       │ │        │  │ • tfidf()│
│  stopword │ │ • tokens     │ │Funções:│  │ • build_ │
│ • stem()  │ │ • vector     │ │add_doc │  │  document│
│ • process │ │              │ │get_pos │  │  _vector │
│           │ │              │ │get_df  │  │ • build_ │
│           │ │              │ │get_term│  │  query_  │
│           │ │              │ │        │  │  vector  │
└───────────┘ └──────────────┘ └────────┘  └──────────┘

             ┌─────────────────────┐
             │    ranker.py        │
             │   (Ranking)         │
             ├─────────────────────┤
             │   class: Ranking    │
             │                     │
             │ Funções:            │
             │ • cosine_similarity │
             │ • rank()            │
             │ • display()         │
             └─────────────────────┘

             ┌─────────────────────┐
             │     query.py        │
             │   (TAD Consulta)    │
             ├─────────────────────┤
             │   class: Query      │
             │                     │
             │ Atributos:          │
             │ • text              │
             │ • tokens            │
             │ • vector            │
             └─────────────────────┘
```

---

## Descrição Detalhada dos Scripts

### 1. **main.py** — Ponto de Entrada

```python
#!/usr/bin/env python3
from src.cli import run

if __name__ == "__main__":
    run()
```

**Responsabilidade**: Iniciar a aplicação.

**O que faz**:
- Importa a função `run()` do módulo `cli.py`
- Executa a função principal quando o script é rodado diretamente

---

### 2. **src/cli.py** — Interface com o Usuário

#### Funções Principais:

##### `select_collection()`
```python
def select_collection():
    # Solicita o caminho da pasta com documentos
    # Valida se o diretório existe
    # Retorna o caminho ou None se inválido
```

**O que faz**:
1. Imprime prompt pedindo o caminho
2. Lê entrada do usuário e remove espaços
3. Valida se o diretório existe
4. Retorna o caminho ou `None` em caso de erro

---

##### `load_documents(collection_path, preprocessor)`
```python
def load_documents(collection_path, preprocessor):
    # Carrega todos os arquivos .txt da pasta
    # Pré-processa cada documento
    # Retorna lista de objetos Document
```

**Passos**:
1. Busca todos os arquivos `.txt` na pasta usando `glob.glob()`
2. Para cada arquivo:
   - Lê o conteúdo com encoding UTF-8
   - Cria um objeto `Document` com ID único, nome e texto
   - Pré-processa o texto usando `preprocessor.process()`
   - Armazena os tokens processados
3. Retorna lista de documentos

**Exemplo**:
```
Arquivo: "texto1.txt"
↓ (leitura)
text = "Python é uma linguagem poderosa..."
↓ (criação de Document)
doc = Document(0, "texto1.txt", text)
↓ (pré-processamento)
doc.tokens = ["python", "lingu", "poder"]  # após remover stopwords e stemming
↓ (adição à lista)
documents = [doc, ...]
```

---

##### `build_index(documents)`
```python
def build_index(documents):
    # Constrói o índice invertido
    # Retorna objeto InvertedIndex
```

**Processo**:
1. Cria novo objeto `InvertedIndex()`
2. Para cada documento:
   - Chama `index.add_document(doc_id, tokens)`
   - Índice armazena cada termo com seus (doc_id, frequência)

**Resultado**:
```
index.index = {
    "python": [(0, 3), (5, 2)],
    "lingu": [(0, 2), (1, 1)],
    "poder": [(0, 1), (2, 3)],
    ...
}
```

---

##### `vectorize_documents(documents, index)`
```python
def vectorize_documents(documents, index):
    # Calcula vetor TF-IDF para cada documento
    # Armazena no atributo 'vector' de cada Document
```

**Processo**:
1. Cria `Vectorizer` com referência ao índice
2. Para cada documento:
   - Calcula vetor TF-IDF: `vectorizer.build_document_vector(doc_id, tokens)`
   - Armazena em `doc.vector`

**Exemplo de Vetor**:
```
doc.vector = {
    "python": 2.15,      # TF-IDF alto
    "lingu": 1.87,
    "poder": 1.45,
    ...
}
```

---

##### `query_loop(vectorizer, documents, preprocessor)`
```python
def query_loop(vectorizer, documents, preprocessor):
    # Loop interativo de consultas
    # Processa, calcula similaridade e exibe ranking
```

**Fluxo por consulta**:
1. Lê consulta do usuário
2. Verifica se é comando de saída ('sair', 'quit', 'exit')
3. Pré-processa a consulta (tokenização, remoção de stopwords, stemming)
4. Constrói vetor TF-IDF da consulta
5. Calcula similaridade com todos os documentos
6. Ordena por relevância
7. Exibe top 10 resultados

---

##### `run()` — Função Principal
```python
def run():
    # Orquestra todo o fluxo do sistema
```

**Sequência de Execução**:

```
1. Imprime banner do sistema
   ============================================================
   SISTEMA DE RECUPERACAO DE INFORMACAO
   Modelo Vetorial (TF-IDF + Cosseno)
   ============================================================

2. Seleciona coleção
   "Caminho da colecao de documentos: "

3. Inicializa pré-processador
   "Inicializando pre-processador..."

4. Carrega documentos
   "Carregando documentos..."
   "  X documento(s) carregado(s)."

5. Constrói índice invertido
   "Construindo indice invertido..."
   "  Y termo(s) unico(s) indexado(s)."

6. Calcula vetores TF-IDF
   "Calculando vetores TF-IDF..."
   "  Vetores calculados com sucesso."

7. Inicia loop de consultas
   "Consulta: " (aguarda entrada)
```

---

### 3. **src/preprocessor.py** — Pré-processamento de Texto

#### Classe `Preprocessor`

**Atributos**:
- `language`: Idioma para processamento (padrão: 'portuguese')
- `stop_words`: Conjunto de palavras ignoráveis
- `stemmer`: Stemmer para reduzir palavras à raiz

**Inicialização**:
```python
def __init__(self, language='portuguese'):
    self.language = language
    self._ensure_nltk_data()  # Baixa dados NLTK se necessário
    self.stop_words = set(stopwords.words(language))
    
    # Usa RSLP para português, Porter para inglês
    if language == 'portuguese':
        self.stemmer = RSLPStemmer()
    else:
        self.stemmer = PorterStemmer()
```

---

#### Método `_ensure_nltk_data()`

**O que faz**:
- Verifica se recursos NLTK (stopwords, RSLP) estão instalados
- Baixa automaticamente se não encontrar
- Garante que o programa funcione na primeira execução

---

#### Método `tokenize(text)`

```python
def tokenize(self, text):
    return re.findall(r'\b\w+\b', text.lower())
```

**O que faz**:
1. Converte texto para minúsculas
2. Extrai apenas palavras (sequências de caracteres alfanuméricos)
3. Remove pontuação e caracteres especiais

**Exemplo**:
```
Input:  "Python é uma LINGUAGEM poderosa! (3.14)"
Output: ["python", "é", "uma", "linguagem", "poderosa", "3", "14"]
```

---

#### Método `remove_stopwords(tokens)`

```python
def remove_stopwords(self, tokens):
    return [t for t in tokens if t not in self.stop_words]
```

**O que faz**:
- Remove palavras comuns que não agregam significado
- Para português, remove: "é", "uma", "de", "o", "a", etc.

**Exemplo**:
```
Input:  ["python", "é", "uma", "linguagem", "poderosa"]
Output: ["python", "linguagem", "poderosa"]

(removidos: "é", "uma")
```

---

#### Método `stem(tokens)`

```python
def stem(self, tokens):
    return [self.stemmer.stem(t) for t in tokens]
```

**O que faz**:
- Reduz palavras à sua forma canônica (raiz)
- RSLP (Removedor de Sufixos da Língua Portuguesa) para português

**Exemplos**:
```
"linguagem"     → "lingu"
"linguagens"    → "lingu"
"programação"   → "program"
"programar"     → "program"
"correndo"      → "corr"
"correr"        → "corr"
```

---

#### Método `process(text)` — Pipeline Completo

```python
def process(self, text):
    tokens = self.tokenize(text)           # Passo 1
    tokens = self.remove_stopwords(tokens)  # Passo 2
    tokens = self.stem(tokens)              # Passo 3
    return tokens
```

**Fluxo Completo**:
```
Input: "As linguagens de programação são ferramentas poderosas"

↓ Tokenização
["as", "linguagens", "de", "programação", "são", "ferramentas", "poderosas"]

↓ Remoção de stopwords
["linguagens", "programação", "ferramentas", "poderosas"]

↓ Stemming
["lingu", "program", "ferram", "poder"]

Output: ["lingu", "program", "ferram", "poder"]
```

---

### 4. **src/document.py** — TAD Documento

#### Classe `Document`

```python
class Document:
    def __init__(self, doc_id, filename, text):
        self.doc_id = doc_id        # ID único (0, 1, 2, ...)
        self.filename = filename    # Nome do arquivo
        self.text = text            # Texto original
        self.tokens = []            # Tokens após pré-processamento
        self.vector = {}            # Vetor TF-IDF
```

**Propósito**: Encapsular um documento com seus metadados e processamentos.

**Atributos**:
- `doc_id`: Identificador único
- `filename`: Nome do arquivo para exibição
- `text`: Conteúdo original (mantido para referência)
- `tokens`: Palavras processadas
- `vector`: Vetor TF-IDF (construído depois)

**Exemplo**:
```python
doc = Document(
    doc_id=0,
    filename="artigo1.txt",
    text="Python é uma linguagem poderosa..."
)
# Após processamento:
doc.tokens = ["python", "lingu", "poder"]
doc.vector = {"python": 2.15, "lingu": 1.87, "poder": 1.45}
```

---

### 5. **src/index.py** — Índice Invertido

#### Classe `InvertedIndex`

```python
class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)      # termo → [(doc_id, freq), ...]
        self.doc_freq = defaultdict(int)    # termo → número de docs
```

---

#### Método `add_document(doc_id, tokens)`

```python
def add_document(self, doc_id, tokens):
    # Contar frequência de cada termo no documento
    freq = defaultdict(int)
    for token in tokens:
        freq[token] += 1
    
    # Armazenar no índice
    for term, count in freq.items():
        self.index[term].append((doc_id, count))
        self.doc_freq[term] += 1
```

**Processo**:
1. Conta frequência de cada termo no documento
2. Para cada termo:
   - Adiciona (doc_id, frequência) à postings list
   - Incrementa document frequency

**Exemplo**:
```
tokens = ["python", "python", "lingu", "poder", "python"]

freq = {"python": 3, "lingu": 1, "poder": 1}

Resultado:
index["python"] = [(0, 3), ...]
index["lingu"] = [(0, 1), ...]
index["poder"] = [(0, 1), ...]

doc_freq["python"] = 1
doc_freq["lingu"] = 1
doc_freq["poder"] = 1
```

---

#### Método `get_postings(term)`
Retorna lista de (doc_id, freq) para um termo.

#### Método `get_document_frequency(term)`
Retorna quantos documentos contêm o termo.

#### Método `get_terms()`
Retorna lista de todos os termos únicos no índice.

---

### 6. **src/vectorizer.py** — Cálculo TF-IDF

#### Classe `Vectorizer`

```python
class Vectorizer:
    def __init__(self, inverted_index, num_docs):
        self.index = inverted_index  # Referência ao índice
        self.num_docs = num_docs     # Total de documentos
```

---

#### Método `tf(freq)` — Term Frequency

```python
def tf(self, freq):
    if freq == 0:
        return 0
    return 1 + math.log10(freq)
```

**Fórmula**: `TF = 1 + log10(frequência)`

**Intuitivamente**:
- Frequência 1: TF = 1 + 0 = 1.0
- Frequência 10: TF = 1 + 1 = 2.0
- Frequência 100: TF = 1 + 2 = 3.0

**Vantagem**: Cresce sublineamente, evitando que termos frequentes dominem.

---

#### Método `idf(term)` — Inverse Document Frequency

```python
def idf(self, term):
    df = self.index.get_document_frequency(term)
    if df == 0:
        return 0
    return 1 + math.log10(self.num_docs / df)
```

**Fórmula**: `IDF = 1 + log10(N / DF)`

**Exemplo com N=1000 documentos**:
```
Termo em 1 documento:    IDF = 1 + log10(1000/1) = 1 + 3 = 4.0 (raro, alto valor)
Termo em 10 documentos:  IDF = 1 + log10(1000/10) = 1 + 2 = 3.0
Termo em 500 documentos: IDF = 1 + log10(1000/500) = 1 + 0.3 = 1.3 (comum, baixo valor)
```

---

#### Método `tfidf(term, freq)`

```python
def tfidf(self, term, freq):
    return self.tf(freq) * self.idf(term)
```

**Resultado Final**: Combina frequência local com importância global.

---

#### Método `build_document_vector(doc_id, tokens)`

```python
def build_document_vector(self, doc_id, tokens):
    vector = {}
    freq = defaultdict(int)
    
    for token in tokens:
        freq[token] += 1
    
    for term, count in freq.items():
        vector[term] = self.tfidf(term, count)
    
    return vector
```

**O que faz**:
1. Conta frequência de cada termo no documento
2. Calcula TF-IDF para cada termo
3. Retorna dicionário {termo: peso}

**Exemplo**:
```
doc_id = 0
tokens = ["python", "python", "lingu", "poder", "python"]
freq = {"python": 3, "lingu": 1, "poder": 1}

TF-IDF calculado:
TF("python") = 1 + log10(3) ≈ 1.477
IDF("python") = 1 + log10(1000/5) ≈ 2.301
TF-IDF("python") ≈ 3.40

Vetor final:
{
    "python": 3.40,
    "lingu": 2.81,
    "poder": 2.81
}
```

---

#### Método `build_query_vector(query_tokens)`

Idêntico ao anterior, mas para consultas. Permite que a consulta seja tratada como um documento para cálculo de similaridade.

---

### 7. **src/query.py** — TAD Consulta

#### Classe `Query`

```python
class Query:
    def __init__(self, text, vector=None):
        self.text = text          # Texto original da consulta
        self.tokens = []          # Tokens processados
        self.vector = vector or {}  # Vetor TF-IDF
```

**Propósito**: Encapsular uma consulta com seus dados.

**Similaridade com `Document`**: Ambos podem ser representados como vetores no espaço de termos.

---

### 8. **src/ranker.py** — Ranking e Similaridade

#### Classe `Ranking`

```python
class Ranking:
    def __init__(self):
        self.results = []  # [(doc_id, filename, score), ...]
```

---

#### Método `cosine_similarity(vec1, vec2)`

```python
def cosine_similarity(self, vec1, vec2):
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    # Produto escalar e norma de vec1
    for term, weight in vec1.items():
        norm1 += weight ** 2
        if term in vec2:
            dot_product += weight * vec2[term]
    
    # Norma de vec2
    for weight in vec2.values():
        norm2 += weight ** 2
    
    # Evita divisão por zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
```

**Fórmula Matemática**:
```
cos(θ) = (vec1 · vec2) / (||vec1|| × ||vec2||)

Onde:
• (vec1 · vec2) = Σ(v1[i] × v2[i])  (produto escalar)
• ||vec1|| = √(Σ v1[i]²)  (norma Euclidiana)
```

**Retorno**: Valor entre 0 e 1
- 0: Vetores ortogonais (sem similaridade)
- 1: Vetores paralelos (máxima similaridade)

**Exemplo**:
```
Query vector:  {"python": 2.0, "lingu": 1.5}
Doc 1 vector:  {"python": 3.5, "lingu": 1.2}
Doc 2 vector:  {"java": 2.0}

Similaridade(query, doc1):
- dot_product = 2.0×3.5 + 1.5×1.2 = 7.0 + 1.8 = 8.8
- ||query|| = √(2.0² + 1.5²) = √(4 + 2.25) = √6.25 = 2.5
- ||doc1|| = √(3.5² + 1.2²) = √(12.25 + 1.44) = √13.69 ≈ 3.7
- cos(θ) = 8.8 / (2.5 × 3.7) ≈ 0.95 (muito similar!)

Similaridade(query, doc2):
- dot_product = 0 (sem termos em comum)
- cos(θ) = 0.0 (não relacionado)
```

---

#### Método `rank(query_vector, documents)`

```python
def rank(self, query_vector, documents):
    self.results = []
    for doc in documents:
        score = self.cosine_similarity(query_vector, doc.vector)
        if score > 0:
            self.results.append((doc.doc_id, doc.filename, score))
    
    self.results.sort(key=lambda x: x[2], reverse=True)
    return self.results
```

**O que faz**:
1. Calcula similaridade entre query e cada documento
2. Mantém apenas documentos com score > 0
3. Ordena em ordem decrescente de similaridade
4. Retorna lista de tuplas (doc_id, filename, score)

---

#### Método `display(top_n=10)`

```python
def display(self, top_n=10):
    if not self.results:
        print("Nenhum documento relevante encontrado.")
        return
    
    print(f"\n{'Posicao':<8} {'Documento':<30} {'Similaridade':<12}")
    print("-" * 50)
    for i, (doc_id, filename, score) in enumerate(self.results[:top_n], 1):
        print(f"{i:<8} {filename:<30} {score:<12.6f}")
```

**O que faz**:
- Exibe os top N resultados em formato tabular
- Mostra posição, nome do arquivo e score de similaridade
- Formata com 6 casas decimais

**Exemplo de Saída**:
```
Posicao  Documento                  Similaridade
--------------------------------------------------
1        artigo_python.txt           0.893451
2        tutorial_programacao.txt    0.726834
3        introducao.txt              0.654219
```

---

## Fluxo de Execução

### Sequência Completa

```
1. Usuário executa: python main.py (ou run.bat)
   ↓

2. main.py → cli.run()
   ↓

3. Banner exibido
   ============================================================
   SISTEMA DE RECUPERACAO DE INFORMACAO
   Modelo Vetorial (TF-IDF + Cosseno)
   ============================================================
   ↓

4. Seleção da Coleção
   "Caminho da colecao de documentos: " → usuário digita "collections"
   ↓

5. Inicialização do Preprocessor
   • Baixa recursos NLTK se necessário
   • Carrega stopwords em português
   • Inicializa stemmer RSLP
   ↓

6. Carregamento de Documentos
   Para cada arquivo .txt em collections/:
   • Lê conteúdo
   • Cria objeto Document
   • Pré-processa (tokenização → remove stopwords → stemming)
   • Armazena tokens
   ↓

7. Construção do Índice Invertido
   Para cada documento:
   • Conta frequência de cada token
   • Armazena em estrutura: termo → [(doc_id, freq), ...]
   ↓

8. Cálculo de Vetores TF-IDF
   Para cada documento:
   • Calcula TF para cada termo
   • Calcula IDF para cada termo
   • Multiplica: TF-IDF = TF × IDF
   • Armazena vetor no objeto Document
   ↓

9. Loop de Consultas
   Enquanto True:
   • "Consulta: " → usuário digita
   • Se "sair": sai do loop
   • Senão:
     - Pré-processa consulta
     - Constrói vetor TF-IDF da consulta
     - Calcula similaridade com cada documento
     - Ordena por similaridade (descendente)
     - Exibe top 10
   ↓

10. Encerramento
    "Encerrando. Ate logo!"
```

### Diagrama de Classes

```
┌─────────────────────┐
│   Document          │
├─────────────────────┤
│ - doc_id: int       │
│ - filename: str     │
│ - text: str         │
│ - tokens: list      │
│ - vector: dict      │
└─────────────────────┘

┌─────────────────────┐
│   Query             │
├─────────────────────┤
│ - text: str         │
│ - tokens: list      │
│ - vector: dict      │
└─────────────────────┘

┌──────────────────────────────┐
│  InvertedIndex               │
├──────────────────────────────┤
│ - index: dict                │
│   termo → [(doc_id, freq)] │
│ - doc_freq: dict             │
│   termo → count_docs         │
├──────────────────────────────┤
│ + add_document()             │
│ + get_postings()             │
│ + get_document_frequency()   │
│ + get_terms()                │
└──────────────────────────────┘

┌──────────────────────────────┐
│  Vectorizer                  │
├──────────────────────────────┤
│ - index: InvertedIndex       │
│ - num_docs: int              │
├──────────────────────────────┤
│ + tf()                       │
│ + idf()                      │
│ + tfidf()                    │
│ + build_document_vector()    │
│ + build_query_vector()       │
└──────────────────────────────┘

┌──────────────────────────────┐
│  Ranking                     │
├──────────────────────────────┤
│ - results: list              │
│   [(doc_id, filename, score)]│
├──────────────────────────────┤
│ + cosine_similarity()        │
│ + rank()                     │
│ + display()                  │
└──────────────────────────────┘

┌──────────────────────────────┐
│  Preprocessor                │
├──────────────────────────────┤
│ - language: str              │
│ - stop_words: set            │
│ - stemmer: object            │
├──────────────────────────────┤
│ + tokenize()                 │
│ + remove_stopwords()         │
│ + stem()                     │
│ + process()                  │
└──────────────────────────────┘
```

---

## Exemplos de Uso

### Exemplo 1: Fluxo Básico

```
Arquivo: articles/machine_learning.txt
Conteúdo: "Machine learning é uma técnica de inteligência artificial 
           que permite computadores aprender com dados."

Processamento:
1. Tokenização:
   ["machine", "learning", "é", "uma", "técnica", "de", ...]

2. Remoção de stopwords:
   ["machine", "learning", "técnica", "inteligência", "artificial", ...]

3. Stemming:
   ["machin", "learn", "tecn", "intel", "artif", ...]

Resultado: tokens = ["machin", "learn", "tecn", "intel", "artif", ...]

Índice:
"machin" → [(0, 1)]
"learn" → [(0, 1)]
"tecn" → [(0, 1)]
...

Vetor TF-IDF do documento:
{
    "machin": 2.15,
    "learn": 2.15,
    "tecn": 2.15,
    ...
}
```

### Exemplo 2: Consulta e Ranking

```
Coleção: 5 documentos
- doc0.txt: "Machine learning"
- doc1.txt: "Inteligência artificial"
- doc2.txt: "Programação em Python"
- doc3.txt: "Machine learning e deep learning"
- doc4.txt: "Análise de dados"

Consulta: "machine learning"
↓ Processamento
Tokens: ["machin", "learn"]

Vetor da consulta:
{
    "machin": 2.15,
    "learn": 2.15
}

Cálculo de Similaridade:
doc0: 0.95 (contém "machine learning")
doc3: 0.98 (contém "machine learning" 2×)
doc1: 0.45 (contém "machine" por coincidência)
doc2: 0.0  (sem termos em comum)
doc4: 0.0  (sem termos em comum)

Ranking Final (ordenado):
1. doc3 (similarity: 0.98)
2. doc0 (similarity: 0.95)
3. doc1 (similarity: 0.45)
```

### Exemplo 3: Impacto do Stemming

```
Sem Stemming:
"computadores", "computador", "computação" → 3 termos distintos

Com Stemming:
"computadores" → "comput"
"computador"  → "comput"
"computação"  → "comput"
Resultado: 1 termo único "comput"

Vantagem: Consulta por "computador" encontra documentos com
          "computadores" e "computação"
```

---

## Cálculos Implementados

### 1. Cálculo do TF (Term Frequency)

**Fórmula**:
```
TF(t, d) = 1 + log10(frequência de t em d)
```

**Implementação** (`src/vectorizer.py`):
```python
def tf(self, freq):
    if freq == 0:
        return 0
    return 1 + math.log10(freq)
```

**Tabela de Valores**:
| Frequência | TF(freq) |
|-----------|----------|
| 0         | 0        |
| 1         | 1.0      |
| 2         | 1.301    |
| 5         | 1.699    |
| 10        | 2.0      |
| 100       | 3.0      |
| 1000      | 4.0      |

---

### 2. Cálculo do IDF (Inverse Document Frequency)

**Fórmula**:
```
IDF(t) = 1 + log10(N / DF(t))
onde N = total de documentos
      DF(t) = número de documentos contendo t
```

**Implementação** (`src/vectorizer.py`):
```python
def idf(self, term):
    df = self.index.get_document_frequency(term)
    if df == 0:
        return 0
    return 1 + math.log10(self.num_docs / df)
```

**Exemplo com N=10000**:
| DF    | Proporção | IDF   | Interpretação          |
|-------|-----------|-------|----------------------|
| 1     | 0.01%     | 5.0   | Termo muito raro     |
| 10    | 0.1%      | 4.0   | Termo raro           |
| 100   | 1%        | 3.0   | Termo pouco comum    |
| 1000  | 10%       | 2.0   | Termo comum          |
| 5000  | 50%       | 1.301 | Termo muito comum    |

---

### 3. Cálculo do TF-IDF

**Fórmula**:
```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Implementação** (`src/vectorizer.py`):
```python
def tfidf(self, term, freq):
    return self.tf(freq) * self.idf(term)
```

**Exemplo Completo**:
```
Coleção: 1000 documentos
Documento 5:
- Contém termo "python": 8 vezes
- Termo aparece em 50 documentos

TF("python", doc5) = 1 + log10(8) = 1 + 0.903 = 1.903
IDF("python") = 1 + log10(1000/50) = 1 + 1.301 = 2.301
TF-IDF = 1.903 × 2.301 = 4.38

Interpretação: "python" é importante neste documento e 
               relativamente raro na coleção
```

---

### 4. Cálculo da Similaridade do Cosseno

**Fórmula**:
```
cos(θ) = (V1 · V2) / (||V1|| × ||V2||)

Onde:
• V1 · V2 = Σ(v1[i] × v2[i])  (produto escalar)
• ||V|| = √(Σ v[i]²)           (norma Euclidiana)
```

**Implementação** (`src/ranker.py`):
```python
def cosine_similarity(self, vec1, vec2):
    dot_product = 0.0
    norm1 = 0.0
    norm2 = 0.0
    
    for term, weight in vec1.items():
        norm1 += weight ** 2
        if term in vec2:
            dot_product += weight * vec2[term]
    
    for weight in vec2.values():
        norm2 += weight ** 2
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
```

**Exemplo**:
```
Query vector:     {"python": 2.0, "code": 1.5}
Document vector:  {"python": 3.0, "code": 1.0, "language": 0.5}

Produto escalar (interseção de termos):
= 2.0×3.0 + 1.5×1.0 = 6.0 + 1.5 = 7.5

Norma da query:
||query|| = √(2.0² + 1.5²) = √(4 + 2.25) = √6.25 = 2.5

Norma do documento:
||doc|| = √(3.0² + 1.0² + 0.5²) = √(9 + 1 + 0.25) = √10.25 ≈ 3.2

Similaridade:
cos(θ) = 7.5 / (2.5 × 3.2) = 7.5 / 8.0 = 0.9375

Interpretação: Query e documento são muito similares (0.9375 ≈ 1.0)
```

---

### 5. Algoritmo Completo de Indexação

**Entrada**: Coleção de N documentos

**Saída**: Índice invertido + Vetores TF-IDF

```
Algoritmo IndexarDocumentos():
  index ← InvertedIndex vazio
  vectorizer ← Vectorizer novo
  documentos ← []
  
  Para cada arquivo em collection_path:
    1. Ler arquivo
    2. Preprocessar texto:
       a. Tokenizar
       b. Remover stopwords
       c. Stemming
    3. Criar Document com tokens
    4. Adicionar ao índice
       - Para cada token com frequência f:
         index[token] ← (doc_id, f)
         doc_freq[token] ← doc_freq[token] + 1
    5. Calcular vetor TF-IDF:
       - Para cada token:
         TF ← 1 + log10(frequência)
         IDF ← 1 + log10(N / doc_freq)
         vetor[token] ← TF × IDF
    6. Armazenar documento com vetor
  
  Retornar (documentos, index, vectorizer)
```

---

### 6. Algoritmo Completo de Busca

**Entrada**: Consulta em texto livre

**Saída**: Ranking de documentos relevantes

```
Algoritmo BuscaDocumentos(query_text):
  1. Preprocessar query_text (igual aos documentos)
     query_tokens ← Preprocessar(query_text)
  
  2. Calcular vetor TF-IDF da consulta
     Para cada token em query_tokens:
       TF ← 1 + log10(frequência no query)
       IDF ← 1 + log10(N / doc_freq[token])
       query_vector[token] ← TF × IDF
  
  3. Calcular similaridade com todos documentos
     resultados ← []
     Para cada documento em documentos:
       score ← CosineSimilarity(query_vector, doc.vector)
       Se score > 0:
         resultados.adicionar((doc.id, doc.nome, score))
  
  4. Ordenar resultados em ordem decrescente
     resultados.sort(por score, descendente)
  
  5. Exibir top-K resultados
     Mostrar primeiros 10 documentos
  
  Retornar resultados
```

---

## Estrutura de Arquivos

```
Modelo-Vetorial/
├── main.py                      # Ponto de entrada
├── run.bat                      # Script para executar (Windows)
├── requirements.txt             # Dependências Python
├── README.md                    # Documentação básica
├── DOCUMENTACAO_COMPLETA.md     # Este arquivo
├── trab.md                      # Requisitos do trabalho
│
├── src/
│   ├── __init__.py             # Inicialização do pacote
│   ├── cli.py                  # Interface com usuário (main)
│   ├── preprocessor.py         # Pré-processamento de texto
│   ├── document.py             # TAD Documento
│   ├── query.py                # TAD Consulta
│   ├── index.py                # Índice invertido
│   ├── vectorizer.py           # Cálculo TF-IDF
│   └── ranker.py               # Ranking e similaridade
│
├── collections/                 # Pasta com documentos (.txt)
│   ├── arquivo1.txt
│   ├── arquivo2.txt
│   └── ...
│
└── venv/                        # Ambiente virtual Python
    └── (dependências instaladas)
```

---

## Execução Passo a Passo

### 1. Ativar Virtual Environment

**Windows (CMD)**:
```cmd
venv\Scripts\activate.bat
```

**Windows (PowerShell)**:
```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/Mac**:
```bash
source venv/bin/activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 3. Preparar Coleção

Criar pasta `collections/` com arquivos `.txt`:
```
collections/
  ├── artigo1.txt
  ├── artigo2.txt
  └── artigo3.txt
```

### 4. Executar Programa

```bash
python main.py
```

Ou usar script batch (Windows):
```cmd
run.bat
```

### 5. Usar o Sistema

```
Caminho da colecao de documentos: collections
  5 documento(s) carregado(s).
  2845 termo(s) unico(s) indexado(s).
  Vetores calculados com sucesso.

--- CONSULTAS ---
Digite 'sair' para encerrar.

Consulta: python e programação
5 documento(s) relevante(s) encontrado(s).

Posicao  Documento                  Similaridade
--------------------------------------------------
1        python_guide.txt            0.876543
2        programming_basics.txt      0.654321
3        intro_to_code.txt           0.543210
4        advanced_topics.txt         0.432109
5        reference_manual.txt        0.321098

Consulta: sair
Encerrando. Ate logo!
```

---

## Análise de Complexidade

### Indexação

| Operação | Complexidade | Explicação |
|----------|-------------|-----------|
| Carregamento | O(D×L) | D docs, L média de palavras por doc |
| Tokenização | O(L) | Lê cada palavra uma vez |
| Remoção stopwords | O(L) | Verifica cada palavra |
| Stemming | O(L×W) | W = média tamanho palavra |
| Construção índice | O(D×U) | U = termos únicos por doc |
| Cálculo TF-IDF | O(D×U) | Para cada termo em cada doc |

**Total**: O(D×L) onde D = docs, L = palavras médias

### Busca

| Operação | Complexidade | Explicação |
|----------|-------------|-----------|
| Pré-processamento query | O(Q) | Q = tamanho da consulta |
| Vetor query | O(U_q) | U_q = termos únicos na query |
| Similaridade por doc | O(U_q) | Apenas termos da query |
| Ranking total | O(D×U_q) | D docs × termos da query |
| Ordenação | O(D log D) | Merge sort ou similar |

**Total**: O(D log D) para ranking

---

## Vantagens e Limitações

### Vantagens

✅ **Simplicidade**: Modelo matemático elegante e bem estabelecido
✅ **Eficiência**: Busca rápida com índice invertido
✅ **Interpretabilidade**: Scores têm significado claro
✅ **Escalabilidade**: Funciona bem com coleções médias
✅ **Flexibilidade**: Fácil adicionar novos documentos

### Limitações

❌ **Semântica**: Não compreende significado, apenas frequência
❌ **Contexto**: Ignora ordem e contexto das palavras
❌ **Polissemia**: Não diferencia significados múltiplos
❌ **Sinonímia**: "carro" e "automóvel" são termos distintos
❌ **Documentos muito grandes**: Vetores de alta dimensionalidade

---

## Melhorias Possíveis

1. **Incorporar n-gramas**: Considerar sequências de palavras
2. **Weighting avançado**: BM25 em vez de TF-IDF
3. **Similaridade semântica**: Word embeddings (Word2Vec, GloVe)
4. **Expansão de query**: Adicionar sinônimos automaticamente
5. **Feedback relevante**: Refinar consulta com feedback do usuário
6. **Cache**: Armazenar vetores calculados
7. **Paralelização**: Processar documentos em paralelo
8. **Interface web**: Frontend para melhor UX

---

## Referências

- **Salton, G., Wong, A., & Yang, C. S. (1975)**: "A vector space model for automatic indexing"
- **Manning, C. D., Raghavan, P., & Schütze, H. (2008)**: "Introduction to Information Retrieval"
- **Robertson, S. (2004)**: "Understanding inverse document frequency"

---

**Documento gerado em**: 13 de maio de 2026
**Versão**: 1.0
**Linguagem**: Python 3.8+
