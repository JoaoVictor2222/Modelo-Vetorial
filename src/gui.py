import glob
import os

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

import difflib

from src.document import Document
from src.index import InvertedIndex
from src.preprocessor import Preprocessor
from src.query import Query
from src.ranker import Ranking
from src.vectorizer import Vectorizer


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.collection_path = None
        self.documents = []
        self.documents_by_id = {}
        self.index = None
        self.vectorizer = None
        self.original_terms = set()
        self.preprocessor = Preprocessor(language="portuguese")
        self.ranker = Ranking()
        self.current_results = []

        self.setWindowTitle("Modelo Vetorial - Interface PySide6")
        self.resize(1000, 1000)

        self._setup_ui()

    def _setup_ui(self):
        container = QWidget()
        container.setStyleSheet(
            "QWidget { background-color: #f8fafc; color: #111827; font-family: 'Segoe UI', Arial, sans-serif; }"
            "QLabel { color: #111827; }"
            "QLineEdit, QTextEdit, QTableWidget { background: white; border: 1px solid #cbd5e1; border-radius: 12px; }"
            "QTableWidget { gridline-color: #e2e8f0; }"
            "QHeaderView::section { background: #e2e8f0; padding: 10px; border: none; font-weight: 700; color: #111827; }"
            "QPushButton { background: #3b82f6; color: white; border: none; border-radius: 12px; padding: 10px 16px; font-weight: 700; }"
            "QPushButton:hover { background: #2563eb; }"
            "QPushButton:disabled { background: #94a3b8; color: #f1f5f9; }"
        )
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        button_layout = QHBoxLayout()
        self.select_button = QPushButton("Selecionar coleção")
        self.select_button.clicked.connect(self.select_collection)
        self.load_button = QPushButton("Carregar coleção")
        self.load_button.setEnabled(False)
        self.load_button.clicked.connect(self.load_collection)
        self.select_button.setFixedHeight(40)
        self.load_button.setFixedHeight(40)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.load_button)
        main_layout.addLayout(button_layout)

        self.collection_label = QLabel("Coleção: nenhuma selecionada")
        self.collection_label.setWordWrap(True)
        self.collection_label.setStyleSheet(
            "font-size: 14px; color: #0f172a; font-weight: 700;"
        )
        main_layout.addWidget(self.collection_label)

        self.status_label = QLabel("Status: aguardando seleção")
        self.status_label.setStyleSheet(
            "color: #475569; margin-bottom: 12px; font-size: 13px;"
        )
        main_layout.addWidget(self.status_label)

        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_frame.setStyleSheet(
            "background: #fafafa; border: 1px solid #d0d7de; border-radius: 10px; padding: 10px;"
        )
        stats_layout = QHBoxLayout(stats_frame)
        self.num_docs_label = QLabel("Documentos carregados: 0")
        self.num_terms_label = QLabel("Termos indexados: 0")
        self.total_tokens_label = QLabel("Tokens processados: 0")
        for label in (self.num_docs_label, self.num_terms_label, self.total_tokens_label):
            label.setStyleSheet(
                "color: #1f2937; font-weight: 700; font-size: 13px; padding: 4px 8px;"
            )
            stats_layout.addWidget(label)
        stats_layout.addStretch(1)
        main_layout.addWidget(stats_frame)

        search_frame = QFrame()
        search_frame.setFrameShape(QFrame.StyledPanel)
        search_frame.setStyleSheet(
            "background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 14px;"
        )
        search_area_layout = QHBoxLayout(search_frame)
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Digite a consulta e clique em Buscar...")
        self.query_input.returnPressed.connect(self.perform_search)
        self.query_input.setFixedHeight(44)
        self.search_button = QPushButton("Buscar")
        self.search_button.setEnabled(False)
        self.search_button.setFixedHeight(44)
        self.search_button.setMinimumWidth(120)
        self.search_button.setStyleSheet(
            "background: #2563eb; color: white; border-radius: 12px; font-weight: 700;"
        )
        self.search_button.clicked.connect(self.perform_search)
        search_area_layout.addWidget(self.query_input)
        search_area_layout.addWidget(self.search_button)
        main_layout.addWidget(search_frame)

        # SPELL CHECK AREA START
        self.suggestion_label = QLabel("")  # spelling suggestion output label
        self.suggestion_label.setStyleSheet(
            "color: #2563eb; font-size: 13px; margin-top: 8px; font-weight: 600;"
        )
        main_layout.addWidget(self.suggestion_label)
        # SPELL CHECK AREA END

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Posição", "Documento", "Similaridade"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setStyleSheet(
            "QTableWidget { background: white; border: 1px solid #cbd5e1; border-radius: 12px; alternate-background-color: #f8fafc; }"
            "QTableWidget::item:selected { background: #dbeafe; color: #111827; }"
        )
        self.results_table.itemSelectionChanged.connect(self.on_result_selected)
        main_layout.addWidget(self.results_table)

        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setPlaceholderText("Selecione um documento nos resultados para visualizar seu conteúdo.")
        self.preview_text.setStyleSheet(
            "QTextEdit { background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px; }"
        )
        main_layout.addWidget(self.preview_text, stretch=1)

        container.setLayout(main_layout)
        self.setCentralWidget(container)

    @Slot()
    def select_collection(self):
        selected = QFileDialog.getExistingDirectory(self, "Selecionar pasta da coleção")
        if selected:
            self.collection_path = selected
            self.collection_label.setText(f"Coleção: {selected}")
            self.status_label.setText("Status: coleção selecionada. Pronto para carregar.")
            self.load_button.setEnabled(True)
            self.search_button.setEnabled(False)
            self.results_table.setRowCount(0)
            self.preview_text.clear()

    @Slot()
    def load_collection(self):
        if not self.collection_path:
            self.show_error("Nenhuma coleção selecionada.")
            return

        text_files = sorted(glob.glob(os.path.join(self.collection_path, "*.txt")))
        if not text_files:
            self.show_error("Nenhum arquivo .txt encontrado na coleção selecionada.")
            return

        self.documents = []
        self.original_terms = set()
        for i, filepath in enumerate(text_files):
            with open(filepath, "r", encoding="utf-8") as stream:
                text = stream.read()
            document = Document(i, os.path.basename(filepath), text)
            raw_tokens = self.preprocessor.remove_stopwords(self.preprocessor.tokenize(text))
            document.tokens = self.preprocessor.process(text)
            self.original_terms.update(raw_tokens)
            self.documents.append(document)

        if not self.documents:
            self.show_error("Nenhum documento válido foi carregado.")
            return

        index = InvertedIndex()
        for document in self.documents:
            index.add_document(document.doc_id, document.tokens)

        self.vectorizer = Vectorizer(index, len(self.documents))
        for document in self.documents:
            document.vector = self.vectorizer.build_document_vector(document.doc_id, document.tokens)

        self.documents_by_id = {doc.doc_id: doc for doc in self.documents}
        self.index = index
        self.search_button.setEnabled(True)
        total_tokens = sum(len(doc.tokens) for doc in self.documents)
        self.num_docs_label.setText(f"Documentos carregados: {len(self.documents)}")
        self.num_terms_label.setText(f"Termos indexados: {len(index.get_terms())}")
        self.total_tokens_label.setText(f"Tokens processados: {total_tokens}")
        self.status_label.setText(f"Status: {len(self.documents)} documento(s) carregado(s). Pronto para buscas.")
        self.results_table.setRowCount(0)
        self.preview_text.clear()

    @Slot()
    def perform_search(self):
        if not self.vectorizer or not self.documents:
            self.show_error("Carregue a coleção antes de realizar a busca.")
            return

        query_text = self.query_input.text().strip()
        if not query_text:
            self.show_error("Digite uma consulta válida.")
            return

        query_tokens = self.preprocessor.process(query_text)
        if not query_tokens:
            self.show_error("Nenhum termo válido encontrado na consulta.")
            return

        query_vector = self.vectorizer.build_query_vector(query_tokens)
        query_obj = Query(query_text, query_vector)
        self.current_results = self.ranker.rank(query_obj.vector, self.documents)
        self.display_results()
        self.update_spelling_suggestion(query_text, query_tokens)  # SPELL CHECK: comment this line to disable

    def update_spelling_suggestion(self, query_text, query_tokens):
        if not self.index or not self.original_terms:
            self.suggestion_label.setText("")
            return

        raw_tokens = self.preprocessor.remove_stopwords(self.preprocessor.tokenize(query_text))
        suggestions = []
        stems = set(self.index.get_terms())

        for raw_token in raw_tokens:
            stemmed = self.preprocessor.stem([raw_token])[0]
            if stemmed not in stems:
                close = difflib.get_close_matches(raw_token, sorted(self.original_terms), n=1, cutoff=0.7)
                if close:
                    suggestions.append((raw_token, close[0]))

        if suggestions:
            suggestion_text = ", ".join(
                f"'{wrong}' → '{correct}'" for wrong, correct in suggestions
            )
            self.suggestion_label.setText(
                f"Sugestão ortográfica: {suggestion_text}. Ajuste a consulta e clique em Buscar novamente."
            )
        else:
            self.suggestion_label.setText("")

    def display_results(self):
        self.results_table.setRowCount(0)
        if not self.current_results:
            self.status_label.setText("Status: nenhuma correspondência encontrada.")
            return

        self.status_label.setText(f"Status: {len(self.current_results)} documento(s) relevante(s) encontrado(s).")
        self.results_table.setRowCount(len(self.current_results))

        for row, (doc_id, filename, score) in enumerate(self.current_results):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self.results_table.setItem(row, 1, QTableWidgetItem(filename))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{score:.6f}"))

        self.results_table.resizeRowsToContents()

    def on_result_selected(self):
        selected_items = self.results_table.selectedItems()
        if not selected_items:
            return

        row = selected_items[0].row()
        doc_id = self.current_results[row][0]
        document = self.documents_by_id.get(doc_id)
        if document:
            self.preview_text.setPlainText(document.text)

    def show_error(self, message):
        QMessageBox.warning(self, "Atenção", message)


def run():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
