# Sistema de Recuperação de Informação — Modelo Vetorial

Implementação de um sistema de recuperação de informações baseado no **modelo vetorial (VSM)** com TF-IDF e similaridade do cosseno.

## Funcionalidades

- Índice invertido com frequência dos termos por documento
- Ponderação TF-IDF (`tf = 1 + log10(freq)`, `idf = log10(N/df)`)
- Similaridade do cosseno entre consulta e documentos
- Ranking ordenado por relevância
- Pré-processamento com remoção de stopwords e stemming (RSLP para português)
- Interface gráfica com PySide6
- Visualização de documentos selecionados
- Busca interna no documento selecionado usando a consulta atual, com destaque e navegação entre ocorrências
- Estatísticas de coleção: documentos carregados, termos indexados e tokens processados
- Sugestão ortográfica simples para consultas

## Estrutura

```
├── main.py                  # Ponto de entrada
├── src/
│   ├── gui.py               # Interface gráfica PySide6
│   ├── cli.py               # Interface de linha de comando
│   ├── preprocessor.py      # Tokenização, stopwords, stemming
│   ├── document.py          # TAD Documento
│   ├── index.py             # Índice invertido
│   ├── vectorizer.py        # Cálculo TF-IDF
│   ├── query.py             # TAD Consulta
│   ├── ranker.py            # Similaridade do cosseno + ranking
└── collections/             # Coleções de documentos (.txt)
```

## Requisitos

- Python 3.8+
- `nltk` (stopwords e stemmer RSLP)
- `numpy`
- `PySide6` (Interface gráfica)

Instalação:

```bash
pip install -r requirements.txt
```

## Uso

### Interface gráfica

```bash
python3 main.py
```

1. Selecione a pasta que contém os documentos `.txt`
2. Carregue a coleção
3. Digite a consulta em português na caixa de busca
4. Selecione um resultado para ver o texto completo do documento

### Interface de linha de comando

```bash
python3 -m src.cli
```

1. Informe o caminho da pasta com os documentos `.txt`
2. Digite consultas em português
3. Digite `sair` para encerrar

## Notas

- Você pode escolher entre a interface gráfica (PySide6) e a interface de comando.
- O `main.py` inicia a interface gráfica por padrão.
- A funcionalidade de corretor ortográfico pode ser desativada comentando a chamada `update_spelling_suggestion` em `src/gui.py`.

## Referência

Trabalho acadêmico baseado no modelo vetorial de recuperação de informação (Salton, Wong & Yang, 1975).
