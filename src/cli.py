import os
import glob

from src.preprocessor import Preprocessor
from src.document import Document
from src.index import InvertedIndex
from src.vectorizer import Vectorizer
from src.query import Query
from src.ranker import Ranking


def select_collection():
    print("\n--- SELECAO DA COLECAO ---")
    path = input("Caminho da colecao de documentos: ").strip()
    if not os.path.isdir(path):
        print("Erro: diretorio nao encontrado.")
        return None
    return path


def load_documents(collection_path, preprocessor):
    doc_files = glob.glob(os.path.join(collection_path, "*.txt"))
    if not doc_files:
        print("Erro: nenhum arquivo .txt encontrado em", collection_path)
        return None

    documents = []
    for i, filepath in enumerate(doc_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        doc = Document(i, os.path.basename(filepath), text)
        doc.tokens = preprocessor.process(text)
        documents.append(doc)

    return documents


def build_index(documents):
    index = InvertedIndex()
    for doc in documents:
        index.add_document(doc.doc_id, doc.tokens)
    return index


def vectorize_documents(documents, index):
    vectorizer = Vectorizer(index, len(documents))
    for doc in documents:
        doc.vector = vectorizer.build_document_vector(doc.doc_id, doc.tokens)
    return vectorizer


def query_loop(vectorizer, documents, preprocessor):
    ranker = Ranking()
    print("\n--- CONSULTAS ---")
    print("Digite 'sair' para encerrar.\n")

    while True:
        query_text = input("Consulta: ").strip()
        if query_text.lower() in ('sair', 'quit', 'exit'):
            break
        if not query_text:
            continue

        query_tokens = preprocessor.process(query_text)
        if not query_tokens:
            print("Nenhum termo valido apos pre-processamento.")
            continue

        query_vector = vectorizer.build_query_vector(query_tokens)
        query_obj = Query(query_text, query_vector)

        results = ranker.rank(query_vector, documents)
        print(f"\n{len(results)} documento(s) relevante(s) encontrado(s).")
        ranker.display(top_n=10)
        print()


def run():
    print("=" * 60)
    print("  SISTEMA DE RECUPERACAO DE INFORMACAO")
    print("  Modelo Vetorial (TF-IDF + Cosseno)")
    print("=" * 60)

    collection_path = select_collection()
    if collection_path is None:
        return

    print("\nInicializando pre-processador...")
    preprocessor = Preprocessor(language='portuguese')

    print("Carregando documentos...")
    documents = load_documents(collection_path, preprocessor)
    if documents is None:
        return
    print(f"  {len(documents)} documento(s) carregado(s).")

    print("Construindo indice invertido...")
    index = build_index(documents)
    print(f"  {len(index.get_terms())} termo(s) unico(s) indexado(s).")

    print("Calculando vetores TF-IDF...")
    vectorizer = vectorize_documents(documents, index)
    print("  Vetores calculados com sucesso.")

    query_loop(vectorizer, documents, preprocessor)
    print("\nEncerrando. Ate logo!")


if __name__ == "__main__":
    run()
