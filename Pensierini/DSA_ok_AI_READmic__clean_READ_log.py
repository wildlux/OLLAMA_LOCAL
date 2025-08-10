import sys
import threading
import cv2
import json
import logging
import time
from datetime import datetime
import numpy as np
import subprocess
import re
import requests
import base64
import wave
from io import BytesIO

from PyQt6.QtCore import (
    QThread, pyqtSignal, QTimer, Qt, QMimeData, QPoint, QObject, QSize,
    QPropertyAnimation, QRect, QEvent, QBuffer, QIODevice
)
from PyQt6.QtGui import QImage, QPixmap, QDrag, QCursor, QIcon, QPainter, QPen, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QSizePolicy,
    QLabel, QPushButton, QHBoxLayout, QComboBox, QLineEdit, QFrame, QGridLayout,
    QDialog, QTextEdit, QTabWidget, QCheckBox, QSlider, QRadioButton,
    QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QStackedWidget,
    QScrollArea, QSpacerItem, QGroupBox, QMenu
)

# Importa la libreria pyttsx3 per una sintesi vocale più robusta
try:
    import pyttsx3
except ImportError:
    logging.error("La libreria 'pyttsx3' non è installata. "
                  "Installala con 'pip install pyttsx3'")
    pyttsx3 = None

# Importa la libreria per il riconoscimento vocale
try:
    import speech_recognition as sr
except ImportError:
    logging.error("La libreria 'speech_recognition' non è installata. "
                  "Per abilitare il riconoscimento vocale, installala con 'pip install SpeechRecognition PyAudio'")
    sr = None

# --- CLASSE THREAD PER LA SINTESI VOCALE (AGGIORNATA) ---
class TTSThread(QThread):
    """
    Un thread dedicato per la lettura vocale del testo utilizzando pyttsx3,
    evitando di bloccare l'interfaccia utente.

    Emette segnali per comunicare lo stato al widget DraggableTextWidget.
    """
    finished_reading = pyqtSignal()
    started_reading = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, text_to_read):
        super().__init__()
        self.text_to_read = text_to_read
        self._is_running = True
        self.engine = None

    def run(self):
        if not pyttsx3:
            self.error_occurred.emit("Libreria 'pyttsx3' non disponibile.")
            return

        try:
            self.engine = pyttsx3.init()
            self.engine.say(self.text_to_read)

            self.started_reading.emit()
            logging.info(f"Lettura in corso: {self.text_to_read}")

            # Il metodo runAndWait blocca l'esecuzione finché l'audio non è terminato.
            # Questo avviene in modo sicuro all'interno del thread.
            self.engine.runAndWait()

            # Se la lettura è stata interrotta, l'engine potrebbe essere fermato.
            # Controlliamo il flag _is_running prima di emettere il segnale.
            if self._is_running:
                self.finished_reading.emit()

        except Exception as e:
            logging.error(f"Errore nel thread di lettura vocale: {e}")
            self.error_occurred.emit(str(e))
        finally:
            # Assicurati di fermare il motore pyttsx3
            if self.engine:
                self.engine.stop()
                self.engine = None

    def stop(self):
        """Interrompe la lettura vocale in modo sicuro."""
        self._is_running = False
        if self.engine:
            self.engine.stop()
        self.wait()
# ---------------------------------------------------------------------------------

class LogEmitter(QObject):
    """Oggetto QObject per emettere segnali di log."""
    new_record = pyqtSignal(str)
    error_occurred = pyqtSignal()

class TextEditLogger(logging.Handler):
    """Handler di logging personalizzato che emette segnali a un QTextEdit."""
    def __init__(self, log_emitter, parent=None):
        super().__init__()
        self.log_emitter = log_emitter
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.log_emitter.new_record.emit(msg)
        if record.levelno >= logging.ERROR:
            self.log_emitter.error_occurred.emit()

face_cascade = None
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logging.info("Classificatore viso caricato correttamente")
except Exception as e:
    logging.error(f"Errore nel caricare il classificatore di cascata: {e}")

class VideoThread(QThread):
    """Thread dedicato per la cattura video e rilevamento."""
    change_pixmap_signal = pyqtSignal(QImage)
    status_signal = pyqtSignal(str)

    def __init__(self, face_detection_enabled=True, hand_detection_enabled=True):
        super().__init__()
        self._run_flag = True
        self.face_detection_enabled = face_detection_enabled
        self.hand_detection_enabled = hand_detection_enabled
        self.hand_color_range = (np.array([0, 100, 100]), np.array([10, 255, 255]))

    def run(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_signal.emit("❌ Webcam non disponibile")
            self._run_flag = False
            return

        while self._run_flag:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)

                if self.face_detection_enabled and face_cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (46, 140, 219), 2)

                if self.hand_detection_enabled:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, self.hand_color_range[0], self.hand_color_range[1])
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        max_contour = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(max_contour) > 5000:
                            (x, y, w, h) = cv2.boundingRect(max_contour)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(frame, "Mano rilevata", (x, y - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(q_image)

        self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class DraggableTextWidget(QFrame):
    """Widget di testo trascinabile con pulsanti di azione."""
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setMinimumHeight(60)
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 #667eea, stop: 1 #764ba2);
                border-radius: 15px;
                margin: 5px;
                color: white;
            }
            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 5px 10px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
            }
        """)

        # --- MODIFICA: Aggiornamento della gestione del thread TTS ---
        self.tts_thread = None
        self.is_reading = False

        layout = QHBoxLayout(self)
        self.text_label = QLabel(text)
        self.text_label.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        self.text_label.setWordWrap(True)
        layout.addWidget(self.text_label, 1)

        button_layout = QVBoxLayout()
        self.read_button = QPushButton("🔊")
        self.read_button.setFixedSize(25, 25)
        self.read_button.setToolTip("Leggi testo")
        self.read_button.clicked.connect(self.toggle_read_text)

        self.delete_button = QPushButton("❌")
        self.delete_button.setFixedSize(25, 25)
        self.delete_button.setToolTip("Elimina")
        self.delete_button.clicked.connect(self.delete_self)

        button_layout.addWidget(self.read_button)
        button_layout.addWidget(self.delete_button)
        layout.addLayout(button_layout)

        self.setAcceptDrops(True)
        self.start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.start_pos:
            distance = (event.pos() - self.start_pos).manhattanLength()
            if distance > QApplication.startDragDistance():
                drag = QDrag(self)
                mime = QMimeData()
                mime.setText(self.text_label.text())
                drag.setMimeData(mime)
                drag.setPixmap(self.grab())
                drag.exec(Qt.DropAction.MoveAction)

    # --- MODIFICA: Utilizza il nuovo thread per la lettura vocale ---
    def toggle_read_text(self):
        """Avvia o ferma la lettura del testo usando il thread."""
        if not self.is_reading:
            self.start_reading()
        else:
            self.stop_reading()

    def start_reading(self):
        if self.tts_thread and self.tts_thread.isRunning():
            return

        self.is_reading = True
        self.read_button.setText("⏹️")
        self.read_button.setStyleSheet("background-color: #e74c3c; color: white;")

        self.tts_thread = TTSThread(self.text_label.text())
        self.tts_thread.started_reading.connect(self.on_reading_started)
        self.tts_thread.finished_reading.connect(self.on_reading_finished)
        self.tts_thread.error_occurred.connect(self.on_reading_error)
        self.tts_thread.start()

    def stop_reading(self):
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop()
        self.is_reading = False
        self.read_button.setText("🔊")
        self.read_button.setStyleSheet("")
        logging.info("Lettura testo interrotta.")

    def on_reading_started(self):
        logging.info("Lettura del testo iniziata.")

    def on_reading_finished(self):
        self.is_reading = False
        self.read_button.setText("🔊")
        self.read_button.setStyleSheet("")
        logging.info("Lettura testo completata.")
        self.tts_thread = None

    def on_reading_error(self, message):
        self.is_reading = False
        self.read_button.setText("🔊")
        self.read_button.setStyleSheet("")
        logging.error(f"Errore durante la lettura vocale: {message}")
        self.tts_thread = None

    def delete_self(self):
        """Rimuove il widget"""
        if self.is_reading:
            self.stop_reading()
        self.setParent(None)
        self.deleteLater()

class ConfigurationDialog(QDialog):
    """Dialog per la configurazione dell'applicazione."""
    def __init__(self, parent=None, settings=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Menu di Configurazione")
        self.setModal(True)
        self.resize(600, 500)
        self.settings = settings or {}

        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        title_label = QLabel("⚙️ Menu di Configurazione")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4a90e2;")
        close_button = QPushButton("❌")
        close_button.setFixedSize(30, 30)
        close_button.clicked.connect(self.accept)

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(close_button)
        layout.addLayout(header_layout)

        self.tab_widget = QTabWidget()
        self.setup_ai_tab()
        self.setup_ui_tab()
        self.setup_gestures_tab()
        layout.addWidget(self.tab_widget)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()

        close_menu_btn = QPushButton("Chiudi Menu")
        close_menu_btn.clicked.connect(self.accept)

        log_btn = QPushButton("Scarica Log Emozioni")
        log_btn.clicked.connect(self.download_log)

        bottom_layout.addWidget(close_menu_btn)
        bottom_layout.addWidget(log_btn)
        layout.addLayout(bottom_layout)

    def setup_ai_tab(self):
        ai_widget = QWidget()
        layout = QVBoxLayout(ai_widget)

        ai_group = QGroupBox("Selezione AI")
        ai_layout = QVBoxLayout(ai_group)
        self.ollama_model_combo = QComboBox()
        self.ollama_model_combo.addItem("Seleziona un modello")
        ai_layout.addWidget(QLabel("Modello Ollama:"))
        ai_layout.addWidget(self.ollama_model_combo)

        test_ollama_btn = QPushButton("Testa Connessione & Modelli")
        test_ollama_btn.clicked.connect(self.test_ollama_connection)
        ai_layout.addWidget(test_ollama_btn)

        self.ollama_status_label = QLabel("Stato: Inattivo")
        self.ollama_status_label.setStyleSheet("color: #4a90e2;")
        ai_layout.addWidget(self.ollama_status_label)
        layout.addWidget(ai_group)

        trigger_group = QGroupBox("Trigger per AI")
        trigger_layout = QVBoxLayout(trigger_group)
        trigger_layout.addWidget(QLabel("Imposta una parola d'ordine per inviare il testo all'AI:"))
        self.ai_trigger_input = QLineEdit("++++")
        trigger_layout.addWidget(self.ai_trigger_input)
        layout.addWidget(trigger_group)

        layout.addStretch()
        self.tab_widget.addTab(ai_widget, "Configurazione AI")

        self.test_ollama_connection()

    def setup_ui_tab(self):
        ui_widget = QWidget()
        layout = QVBoxLayout(ui_widget)

        layout.addWidget(QLabel("Tema & Colori Cursore"))
        layout.addWidget(QLabel("Scegli il tema dell'interfaccia:"))

        theme_layout = QHBoxLayout()
        self.current_theme_btn = QPushButton("Tema Attuale")
        self.current_theme_btn.setCheckable(True)
        self.current_theme_btn.setChecked(True)
        self.school_theme_btn = QPushButton("Tema Scuola")
        self.school_theme_btn.setCheckable(True)

        theme_layout.addWidget(self.current_theme_btn)
        theme_layout.addWidget(self.school_theme_btn)
        layout.addLayout(theme_layout)

        colors_layout = QHBoxLayout()
        colors_layout.addWidget(QLabel("Colore mano sinistra:"))
        colors_layout.addWidget(QPushButton("—"))
        colors_layout.addWidget(QLabel("Colore mano destra:"))
        colors_layout.addWidget(QPushButton("—"))
        colors_layout.addWidget(QLabel("Colore viso:"))
        colors_layout.addWidget(QPushButton("—"))
        layout.addLayout(colors_layout)

        layout.addWidget(QLabel("Layout Elementi"))
        layout.addWidget(QLabel("Dimensioni degli elementi contenuti:"))

        self.size_combo = QComboBox()
        self.size_combo.addItems(["Adatta al testo", "Piccolo", "Medio", "Grande"])
        layout.addWidget(self.size_combo)

        layout.addWidget(QLabel("Visualizzazione Elementi"))
        layout.addWidget(QLabel("Scegli come visualizzare testo e icone:"))

        self.visualization_combo = QComboBox()
        self.visualization_combo.addItems(["Testo con Icona", "Solo Testo", "Solo Icona"])
        layout.addWidget(self.visualization_combo)

        layout.addWidget(QLabel("Posizione Icone"))
        layout.addWidget(QLabel("Scegli la posizione delle icone negli elementi:"))

        self.icon_position_combo = QComboBox()
        self.icon_position_combo.addItems(["In alto", "In basso", "A sinistra", "A destra"])
        layout.addWidget(self.icon_position_combo)

        layout.addStretch()
        self.tab_widget.addTab(ui_widget, "Comportamento & UI")

    def setup_gestures_tab(self):
        gestures_widget = QWidget()
        layout = QVBoxLayout(gestures_widget)

        layout.addWidget(QLabel("Riconoscimento Vocale"))
        layout.addWidget(QLabel("Seleziona la lingua per il microfono:"))

        self.language_combo = QComboBox()
        self.language_combo.addItems(["Italiano", "English", "Français", "Deutsch"])
        layout.addWidget(self.language_combo)

        layout.addWidget(QLabel("Riconoscimento Facciale"))
        face_layout = QVBoxLayout()
        face_layout.addWidget(QLabel("Attiva la funzione d'emergenza \"genitore empatico\":"))

        self.face_recognition_cb = QCheckBox("Abilita")
        face_layout.addWidget(self.face_recognition_cb)
        layout.addLayout(face_layout)

        layout.addWidget(QLabel("Dispositivo Esterno"))
        layout.addWidget(QLabel("Collega Arduino o altri dispositivi via USB."))

        connect_device_btn = QPushButton("🔗 Collega Dispositivo Esterno")
        layout.addWidget(connect_device_btn)

        layout.addWidget(QLabel("Feedback Sonoro"))
        layout.addWidget(QLabel("Attiva/disattiva i suoni per i gesti:"))

        self.sound_cb = QCheckBox("Abilita Suoni")
        self.sound_cb.setChecked(True)
        layout.addWidget(self.sound_cb)

        layout.addWidget(QLabel("Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        layout.addWidget(self.volume_slider)

        layout.addWidget(QLabel("Gesti Mano"))
        layout.addWidget(QLabel("Timeout per la selezione (in ms):"))

        self.timeout_input = QLineEdit("500")
        layout.addWidget(self.timeout_input)

        layout.addStretch()
        self.tab_widget.addTab(gestures_widget, "Gesti & Suoni")

    def test_ollama_connection(self):
        """Testa la connessione a Ollama e recupera i modelli."""
        self.ollama_status_label.setText("Stato: Test in corso...")
        self.ollama_model_combo.clear()
        try:
            models = self.get_ollama_models()
            if models:
                self.ollama_status_label.setText("Stato: ✅ Connesso a Ollama")
                self.ollama_model_combo.addItems(models)
                self.ollama_model_combo.setCurrentIndex(0)
            else:
                self.ollama_status_label.setText("Stato: ⚠️ Connesso, ma nessun modello trovato")
        except requests.exceptions.ConnectionError:
            self.ollama_status_label.setText("Stato: ❌ Errore di connessione")
            QMessageBox.critical(self, "Errore di Connessione",
                                 "Impossibile connettersi al server Ollama. Assicurati che sia in esecuzione.")
        except Exception as e:
            self.ollama_status_label.setText(f"Stato: ❌ Errore - {e}")
            logging.error(f"Errore Ollama: {e}")
            QMessageBox.critical(self, "Errore", f"Si è verificato un errore: {e}")

    def get_ollama_models(self):
        """Recupera la lista dei modelli da Ollama."""
        try:
            response = requests.get('http://localhost:11434/api/tags')
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        except Exception as e:
            logging.error(f"Impossibile recuperare i modelli da Ollama: {e}")
            raise

    def download_log(self):
        """Scarica il log delle emozioni"""
        logging.info("Download log emozioni richiesto")
        QMessageBox.information(self, "Download Log", "Funzionalità non ancora implementata.")

    def load_settings(self):
        """Carica le impostazioni salvate"""
        pass

    def get_settings(self):
        """Restituisce le impostazioni correnti."""
        return {
            'ai_trigger': self.ai_trigger_input.text(),
            'ollama_model': self.ollama_model_combo.currentText(),
            'language': self.language_combo.currentText(),
            'face_recognition': self.face_recognition_cb.isChecked(),
            'sound_enabled': self.sound_cb.isChecked(),
            'volume': self.volume_slider.value(),
            'timeout': self.timeout_input.text(),
            'theme': 'current' if self.current_theme_btn.isChecked() else 'school',
            'element_size': self.size_combo.currentText(),
            'visualization': self.visualization_combo.currentText(),
            'icon_position': self.icon_position_combo.currentText()
        }

class ScrollablePanel(QScrollArea):
    """Pannello scrollabile che accetta drop."""
    def __init__(self, title, color_class):
        super().__init__()
        self.setWidgetResizable(True)
        self.setAcceptDrops(True)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 10px; color: #4a90e2;")
        self.layout.addWidget(title_label)

        self.setWidget(self.container)
        self.setStyleSheet(f"""
            QScrollArea {{
                background-color: #f5f7fa;
                border: 3px solid {self.get_border_color(color_class)};
                border-radius: 15px;
                margin: 5px;
            }}
            QScrollBar:vertical {{
                background: #c3cfe2;
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: #4a90e2;
                border-radius: 6px;
            }}
        """)

    def get_border_color(self, color_class):
        colors = {
            'blue': '#3498db',
            'red': '#e74c3c',
            'green': '#2ecc71'
        }
        return colors.get(color_class, '#bdc3c7')

    def add_widget(self, widget):
        self.layout.addWidget(widget)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        text = event.mimeData().text()
        if text:
            new_widget = DraggableTextWidget(text)
            self.add_widget(new_widget)
            event.acceptProposedAction()

class MainWindow(QMainWindow):
    """Finestra principale dell'applicazione."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Assistente per Dislessia")
        self.setGeometry(100, 100, 1400, 800)

        self.settings = {}
        self.is_listening = False
        self.voice_thread = None

        self.setup_ui()
        self.setup_video_thread()
        self.setStyleSheet(self.get_main_stylesheet())

        self.ollama_thread = None

        self.init_settings()

    def init_settings(self):
        """Inizializza le impostazioni e tenta di connettersi a Ollama all'avvio."""
        dialog = ConfigurationDialog(self, self.settings)
        self.settings = dialog.get_settings()
        logging.info("Impostazioni iniziali caricate.")

    def setup_ui(self):
        """Configura l'interfaccia utente."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setScaledContents(True)
        self.video_label.setMinimumHeight(200)
        main_layout.addWidget(self.video_label)

        overlay_widget = QWidget()
        overlay_widget.setStyleSheet("background-color: transparent;")
        overlay_layout = QVBoxLayout(overlay_widget)
        overlay_layout.setContentsMargins(20, 20, 20, 20)

        self.setup_top_bar(overlay_layout)
        self.setup_central_panels(overlay_layout)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.hide()
        self.log_widget.setStyleSheet("""
            QTextEdit {
                background-color: #333;
                color: #fff;
                border: 2px solid #555;
                border-radius: 8px;
                padding: 10px;
                font-family: monospace;
            }
        """)
        overlay_layout.addWidget(self.log_widget)

        self.setup_bottom_bar(overlay_layout)

        overlay_widget.setParent(central_widget)
        overlay_widget.setGeometry(central_widget.rect())
        central_widget.installEventFilter(self)

        self.log_emitter = LogEmitter()
        self.log_handler = TextEditLogger(self.log_emitter, self.log_widget)
        self.log_emitter.new_record.connect(self.log_widget.append)
        self.log_emitter.error_occurred.connect(self.toggle_log_visibility)
        logging.root.addHandler(self.log_handler)
        logging.root.setLevel(logging.INFO)

    def setup_top_bar(self, parent_layout):
        """Configura la barra superiore."""
        top_layout = QHBoxLayout()

        self.options_btn = QPushButton("⚙️ Opzioni")
        self.options_btn.clicked.connect(self.show_options)
        top_layout.addWidget(self.options_btn)

        top_layout.addStretch()

        title_label = QLabel("Assistente per Dislessia")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        top_layout.addWidget(title_label)

        top_layout.addStretch()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: white; font-weight: bold;")
        top_layout.addWidget(self.status_label)

        self.save_btn = QPushButton("💾 Salva")
        top_layout.addWidget(self.save_btn)

        parent_layout.addLayout(top_layout)

    def setup_central_panels(self, parent_layout):
        """Configura i pannelli centrali."""
        panels_layout = QHBoxLayout()
        panels_layout.setSpacing(10)

        self.contents_panel = ScrollablePanel("📝 Contenuti pensieri creativi (A)", "blue")
        panels_layout.addWidget(self.contents_panel, 1)

        self.work_area_panel = ScrollablePanel("🎯 Area di Lavoro (B)", "red")
        placeholder_label = QLabel("Trascina gli elementi qui per lavorarci")
        placeholder_label.setStyleSheet("color: #888; font-style: italic; margin: 20px;")
        placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.work_area_panel.add_widget(placeholder_label)
        panels_layout.addWidget(self.work_area_panel, 1)

        self.ai_panel = ScrollablePanel("📋 Dettagli & Risultati Artificial Intelligence (C)", "green")
        self.ai_results_text = QTextEdit()
        self.ai_results_text.setReadOnly(True)
        self.ai_results_text.setPlainText("Le risposte dell'AI appariranno qui.")
        self.ai_results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f5f7fa;
                border: none;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
            }
        """)
        self.ai_panel.add_widget(self.ai_results_text)
        panels_layout.addWidget(self.ai_panel, 1)

        parent_layout.addLayout(panels_layout)

    def setup_bottom_bar(self, parent_layout):
        """Configura la barra inferiore."""
        bottom_layout = QHBoxLayout()

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Inserisci il tuo testo qui...")
        self.text_input.returnPressed.connect(self.add_text)
        self.text_input.setStyleSheet("""
            QLineEdit {
                background-color: #fff;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                color: #333;
            }
            QLineEdit:focus {
                border: 2px solid #4a90e2;
                background-color: white;
            }
        """)
        bottom_layout.addWidget(self.text_input, 1)

        self.add_btn = QPushButton("➕ Aggiungi")
        self.add_btn.clicked.connect(lambda: self.add_text())

        self.ai_btn = QPushButton("🧠 AI")
        self.ai_btn.clicked.connect(self.send_to_ai)

        voice_icon = QIcon(self.get_svg_icon("M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3.99-3.98 5.75-5.3 5.75S6.7 14.99 6.7 11H5c0 4.14 3.36 7.48 7.48 7.48V21h-3v2h6v-2h-3v-2.52c4.14 0 7.48-3.34 7.48-7.48h-2.18z"))
        recording_icon = QIcon(self.get_svg_icon("M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"))
        self.voice_btn = QPushButton("Registra Voce")
        self.voice_btn.setObjectName("voiceButton")
        self.voice_btn.setIcon(voice_icon)
        self.voice_btn.clicked.connect(self.toggle_voice_input)

        self.hands_btn = QPushButton("✋ Mani")
        self.face_btn = QPushButton("😊 Faccia")
        self.clean_btn = QPushButton("🧹 Pulisci")
        self.clean_btn.clicked.connect(self.clean_input)

        self.log_btn = QPushButton("📊 Mostra Log")
        self.log_btn.clicked.connect(self.toggle_log_visibility)

        for btn in [self.add_btn, self.ai_btn, self.voice_btn,
                    self.hands_btn, self.face_btn, self.clean_btn, self.log_btn]:
            bottom_layout.addWidget(btn)

        parent_layout.addLayout(bottom_layout)

    def get_svg_icon(self, path_data):
        """Genera un'icona da dati SVG."""
        svg_content = f"""
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="#fff">
            <path d="{path_data}"/>
        </svg>
        """
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        buffer.write(svg_content.encode('utf-8'))
        buffer.close()
        svg_bytes = buffer.data()
        pixmap = QPixmap()
        pixmap.loadFromData(svg_bytes)
        return pixmap

    def setup_video_thread(self):
        """Inizializza il thread video."""
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_video)
        self.video_thread.status_signal.connect(self.update_status)
        self.video_thread.start()

    def eventFilter(self, obj, event):
        """Gestisce il ridimensionamento dell'overlay."""
        if obj is self.centralWidget() and event.type() == QEvent.Type.Resize:
            for child in obj.findChildren(QWidget):
                if child.parent() == obj and hasattr(child, 'setGeometry'):
                    child.setGeometry(obj.rect())
        return super().eventFilter(obj, event)

    def toggle_log_visibility(self):
        """Mostra o nasconde il pannello di log."""
        if self.log_widget.isVisible():
            self.log_widget.hide()
            self.log_btn.setText("📊 Mostra Log")
        else:
            self.log_widget.show()
            self.log_btn.setText("📊 Nascondi Log")

    def show_options(self):
        """Mostra il dialog delle opzioni."""
        dialog = ConfigurationDialog(self, self.settings)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.settings = dialog.get_settings()
            logging.info("Impostazioni aggiornate")

    def add_text(self, text=None):
        """Aggiunge testo al pannello contenuti. Gestisce sia l'invio da tastiera che da pulsante."""
        if text is None or isinstance(text, bool):
            text = self.text_input.text().strip()
            if not text:
                return
            self.text_input.clear()

        widget = DraggableTextWidget(text)
        self.contents_panel.add_widget(widget)
        logging.info(f"Testo aggiunto: {text}")

    def send_to_ai(self):
        """Invia il testo all'AI e gestisce la risposta in un thread."""
        input_text = self.text_input.text().strip()
        if not input_text:
            logging.warning("Inserisci del testo per inviarlo all'AI.")
            QMessageBox.warning(self, "Attenzione", "Inserisci del testo per inviarlo all'AI.")
            return

        ollama_model = self.settings.get('ollama_model')
        if not ollama_model or ollama_model == "Seleziona un modello":
            logging.error("Nessuno modello Ollama selezionato.")
            QMessageBox.critical(self, "Errore", "Nessuno modello Ollama selezionato. Controlla le opzioni.")
            return

        self.update_status(f"🧠 AI in elaborazione con {ollama_model}...")
        self.ai_btn.setEnabled(False)
        self.add_btn.setEnabled(False)

        self.ollama_thread = OllamaThread(input_text, ollama_model)
        self.ollama_thread.ollama_response.connect(self.handle_ollama_response)
        self.ollama_thread.ollama_error.connect(self.handle_ollama_error)
        self.ollama_thread.start()

    def handle_ollama_response(self, response_text):
        """Gestisce la risposta di Ollama."""
        self.update_status("✅ Risposta AI ricevuta.")
        self.ai_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.add_text(response_text)
        self.ai_results_text.setText(response_text)
        logging.info("Risposta Ollama processata con successo.")

    def handle_ollama_error(self, error_message):
        """Gestisce gli errori di Ollama."""
        self.update_status("❌ Errore nella richiesta AI.")
        self.ai_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.ai_results_text.setText(f"Errore: {error_message}")
        logging.error(f"Errore nella richiesta Ollama: {error_message}")
        QMessageBox.critical(self, "Errore Ollama", error_message)

    def toggle_voice_input(self, checked=False):
        """Inizia o ferma la registrazione vocale."""
        if not sr:
            logging.error("Libreria 'speech_recognition' non disponibile.")
            QMessageBox.critical(self, "Errore", "La libreria 'speech_recognition' non è disponibile.")
            return

        if not self.is_listening:
            self.is_listening = True
            self.voice_btn.setText("Registrazione in corso...")
            voice_icon = QIcon(self.get_svg_icon("M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"))
            self.voice_btn.setIcon(voice_icon)
            self.voice_btn.setStyleSheet("""
                #voiceButton {
                    background-color: #e74c3c;
                    color: white;
                    border-radius: 12px;
                    padding: 8px 16px;
                }
            """)
            self.update_status("🎤 Registrazione in corso...")

            self.voice_thread = VoiceRecognitionThread(self.settings.get('language', 'Italiano'))
            self.voice_thread.recognized_text.connect(self.handle_recognized_text)
            self.voice_thread.recognition_error.connect(self.handle_recognition_error)
            self.voice_thread.finished.connect(self.stop_voice_input)
            self.voice_thread.start()
        else:
            if self.voice_thread and self.voice_thread.isRunning():
                self.voice_thread.stop()
            self.stop_voice_input()

    def handle_recognized_text(self, text):
        """Gestisce il testo riconosciuto dal thread."""
        current_text = self.text_input.text()
        if current_text:
            self.text_input.setText(f"{current_text} {text}")
        else:
            self.text_input.setText(text)
        self.update_status("✅ Testo riconosciuto.")
        logging.info(f"Testo riconosciuto vocalmente: {text}")

    def handle_recognition_error(self, message):
        """Gestisce gli errori di riconoscimento vocale."""
        QMessageBox.warning(self, "Errore Riconoscimento Vocale", message)
        self.update_status(f"❌ Errore: {message}")
        logging.error(f"Errore di riconoscimento vocale: {message}")

    def stop_voice_input(self):
        """Ferma la registrazione vocale e ripristina il pulsante."""
        self.is_listening = False
        self.voice_btn.setText("Registra Voce")
        voice_icon = QIcon(self.get_svg_icon("M12 14c1.66 0 2.99-1.34 2.99-3L15 5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5.3-3c0 3.99-3.98 5.75-5.3 5.75S6.7 14.99 6.7 11H5c0 4.14 3.36 7.48 7.48 7.48V21h-3v2h6v-2h-3v-2.52c4.14 0 7.48-3.34 7.48-7.48h-2.18z"))
        self.voice_btn.setIcon(voice_icon)
        self.voice_btn.setStyleSheet("")
        self.update_status("✅ Sistema attivo - Mostra le mani o il viso")

    def clean_input(self):
        """Pulisce l'input di testo."""
        self.text_input.clear()
        logging.info("Campo di input testo pulito.")

    def update_video(self, q_image):
        """Aggiorna il frame video."""
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap)

    def update_status(self, message):
        """Aggiorna il messaggio di stato."""
        self.status_label.setText(message)
        logging.info(f"Status: {message}")

    def get_main_stylesheet(self):
        """Restituisce il foglio di stile principale."""
        return """
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 #667eea, stop: 1 #764ba2);
            }

            QPushButton {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 12px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12px;
            }

            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.3);
                border: 2px solid rgba(255, 255, 255, 0.5);
            }

            QPushButton:pressed {
                background-color: rgba(255, 255, 255, 0.4);
            }

            QPushButton:disabled {
                background-color: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.5);
                border: 2px solid rgba(255, 255, 255, 0.2);
            }

            QLineEdit {
                background-color: #fff;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 8px 12px;
                font-size: 14px;
                color: #333;
            }

            QLineEdit:focus {
                border: 2px solid #4a90e2;
                background-color: white;
            }

            QLabel {
                color: white;
            }

            QDialog QLabel {
                color: #333;
            }

            QDialog QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 5px 10px;
            }

            QDialog QPushButton:hover {
                background-color: #3b82f6;
            }

            QComboBox {
                background-color: white;
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 5px;
                min-width: 100px;
                color: #333;
            }

            QComboBox::drop-down {
                border: none;
            }

            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTcgMTBMMTIgMTVMMTcgMTBaIiBmaWxsPSIjNDU0NTQ1Ii8+Cjwvc3ZnPg==);
                width: 16px;
                height: 16px;
                margin-right: 5px;
            }

            QTextEdit {
                background-color: #f5f7fa;
                border: 2px solid #ccc;
                border-radius: 8px;
                padding: 8px;
                color: #333;
            }

            QCheckBox {
                color: #333;
                font-weight: bold;
            }

            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #4a90e2;
                background-color: white;
            }

            QCheckBox::indicator:checked {
                background-color: #4a90e2;
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iOCIgdmlld0JveD0iMCAwIDEwIDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9MTAuNSAyLjVMNC41IDguNUwyLjUgNi41IiBzdHJva2U9IiMxNDEyMTIiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPg==);
            }

            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
                background: #4a90e2;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::add-page:horizontal {
                background: #fff;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: #4a90e2;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin-top: -2px;
                margin-bottom: -2px;
                border-radius: 3px;
            }

            QTabWidget::pane {
                border: 2px solid #4a90e2;
                border-radius: 8px;
                background-color: white;
            }

            QTabBar::tab {
                background-color: #f5f7fa;
                border: 2px solid #4a90e2;
                border-bottom-color: transparent;
                border-radius: 8px 8px 0px 0px;
                min-width: 120px;
                padding: 8px 16px;
                margin-right: 2px;
                font-weight: bold;
                color: #4a90e2;
            }

            QTabBar::tab:selected {
                background-color: #4a90e2;
                color: white;
            }

            QTabBar::tab:hover {
                background-color: rgba(74, 144, 226, 0.3);
            }

            QDialog {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 #f5f7fa, stop: 1 #c3cfe2);
            }
        """

    def closeEvent(self, event):
        """Gestisce la chiusura dell'applicazione."""
        self.video_thread.stop()
        if self.voice_thread and self.voice_thread.isRunning():
            self.voice_thread.stop()
        event.accept()

class OllamaThread(QThread):
    """Thread per la gestione delle chiamate API a Ollama."""
    ollama_response = pyqtSignal(str)
    ollama_error = pyqtSignal(str)

    def __init__(self, prompt, model):
        super().__init__()
        self.prompt = prompt
        self.model = model

    def run(self):
        try:
            logging.info(f"Invio prompt a Ollama. Modello: {self.model}, Prompt: {self.prompt}")
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": self.prompt,
                "stream": False
            }

            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            full_response = data.get("response", "Nessuna risposta ricevuta.")

            self.ollama_response.emit(full_response.strip())
            logging.info("Risposta Ollama ricevuta.")

        except requests.exceptions.ConnectionError:
            logging.error("Errore di connessione: Il server Ollama non è raggiungibile.")
            self.ollama_error.emit("Errore di connessione: Il server Ollama non è raggiungibile. Assicurati che sia in esecuzione.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Errore nella richiesta Ollama: {e}")
            self.ollama_error.emit(f"Errore nella richiesta Ollama: {e}")
        except Exception as e:
            logging.error(f"Si è verificato un errore inaspettato: {e}")
            self.ollama_error.emit(f"Si è verificato un errore inaspettato: {e}")

class VoiceRecognitionThread(QThread):
    """Thread per il riconoscimento vocale asincrono."""
    recognized_text = pyqtSignal(str)
    recognition_error = pyqtSignal(str)

    def __init__(self, lang_setting):
        super().__init__()
        self._running = True
        self.recognizer = sr.Recognizer()

        lang_map = {
            'Italiano': 'it-IT',
            'English': 'en-US',
            'Français': 'fr-FR',
            'Deutsch': 'de-DE'
        }
        self.lang_code = lang_map.get(lang_setting, 'it-IT')

    def run(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            logging.info("In ascolto per il riconoscimento vocale...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                if not self._running:
                    return

                logging.info("Riconoscimento in corso...")
                text = self.recognizer.recognize_google(audio, language=self.lang_code)
                self.recognized_text.emit(text)

            except sr.WaitTimeoutError:
                logging.warning("Tempo di attesa scaduto per il riconoscimento vocale.")
                self.recognition_error.emit("Tempo di attesa scaduto. Nessun input vocale ricevuto.")
            except sr.UnknownValueError:
                logging.warning("Impossibile riconoscere il testo dal segnale audio.")
                self.recognition_error.emit("Impossibile riconoscere il testo. Riprova.")
            except sr.RequestError as e:
                logging.error(f"Errore dal servizio di riconoscimento vocale: {e}")
                self.recognition_error.emit(f"Errore dal servizio di riconoscimento vocale; {e}")
            except Exception as e:
                logging.error(f"Si è verificato un errore inaspettato nel riconoscimento vocale: {e}")
                self.recognition_error.emit(f"Si è verificato un errore inaspettato: {e}")

    def stop(self):
        self._running = False
        self.wait()

def main():
    """Funzione principale."""
    app = QApplication(sys.argv)
    app.setApplicationName("Assistente per Dislessia")
    app.setOrganizationName("DSA Helper")

    try:
        app.setWindowIcon(QIcon("icon.png"))
    except:
        pass

    window = MainWindow()
    window.show()

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
