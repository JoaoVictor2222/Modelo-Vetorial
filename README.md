# Sistema de Recuperação de Informação — Modelo Vetorial

Implementação de um sistema de recuperação de informações baseado no **modelo vetorial (VSM)** com TF-IDF e similaridade do cosseno.

## Funcionalidades

- Índice invertido com frequência dos termos por documento
- Ponderação TF-IDF (`tf = 1 + log10(freq)`, `idf = log10(N/df)`)
- Similaridade do cosseno entre consulta e documentos
- Ranking ordenado por relevância
- Pré-processamento com remoção de stopwords e stemming (RSLP para português)
- Interface de linha de comando

## Estrutura

```
├── main.py                  # Ponto de entrada
├── src/
│   ├── preprocessor.py      # Tokenização, stopwords, stemming
│   ├── document.py          # TAD Documento
│   ├── index.py             # Índice invertido
│   ├── vectorizer.py        # Cálculo TF-IDF
│   ├── query.py             # TAD Consulta
│   ├── ranker.py            # Similaridade do cosseno + ranking
│   └── cli.py               # Interface com o usuário
└── collections/             # Coleções de documentos (.txt)
```

## Requisitos

- Python 3.8+
- `nltk` (stopwords e stemmer RSLP)
- `numpy`

Instalação:

```bash
pip install -r requirements.txt
```

## Uso

```bash
python3 main.py
```

1. Informe o caminho da pasta com os documentos `.txt`
2. Digite consultas em português
3. Digite `sair` para encerrar

## Referência

Trabalho acadêmico baseado no modelo vetorial de recuperação de informação (Salton, Wong & Yang, 1975).
