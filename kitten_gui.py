import os
import sys
import tempfile
import time

try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    os.add_dll_directory(r"C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.9\x64")
except FileNotFoundError:
    pass

from kittentts import KittenTTS
import soundfile as sf
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


class ModelLoader(QThread):
    loaded = pyqtSignal(object, list, list, list, str)
    failed = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            model = KittenTTS(self.model_name)
            providers = model.model.session.get_providers()
            aliases = list(getattr(model.model, "voice_aliases", {}).keys())
            voices = list(getattr(model.model, "available_voices", []))
            self.loaded.emit(model, providers, aliases, voices, self.model_name)
        except Exception as error:
            self.failed.emit(str(error))


class GenerateWorker(QThread):
    finished = pyqtSignal(str, float, float)
    failed = pyqtSignal(str)

    def __init__(self, model, text, voice, speed, output_path):
        super().__init__()
        self.model = model
        self.text = text
        self.voice = voice
        self.speed = speed
        self.output_path = output_path

    def run(self):
        try:
            start_time = time.perf_counter()
            audio = self.model.generate(text=self.text, voice=self.voice, speed=self.speed)
            generation_seconds = time.perf_counter() - start_time
            audio_seconds = float(len(audio)) / 24000.0
            sf.write(self.output_path, audio, 24000)
            self.finished.emit(self.output_path, generation_seconds, audio_seconds)
        except Exception as error:
            self.failed.emit(str(error))


class KittenTTSWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KittenTTS")
        self.setMinimumWidth(720)

        self.model = None
        self.output_path = os.path.join(tempfile.gettempdir(), "kitten_tts_preview.wav")
        self.player = QMediaPlayer(self)

        self.status_label = QLabel("Loading model...")
        self.providers_label = QLabel("")
        self.metrics_label = QLabel("")
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to synthesize")
        self.voice_selector = QComboBox()
        self.speed_input = QDoubleSpinBox()
        self.speed_input.setRange(0.5, 2.0)
        self.speed_input.setSingleStep(0.1)
        self.speed_input.setValue(1.0)
        self.generate_button = QPushButton("Generate and Play")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_audio)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Voice"))
        controls_layout.addWidget(self.voice_selector)
        controls_layout.addWidget(QLabel("Speed"))
        controls_layout.addWidget(self.speed_input)

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.providers_label)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.text_input)
        layout.addLayout(controls_layout)
        layout.addWidget(self.generate_button)
        self.setLayout(layout)

        self.loader = ModelLoader("KittenML/kitten-tts-mini-0.8")
        self.loader.loaded.connect(self.on_model_loaded)
        self.loader.failed.connect(self.on_model_failed)
        self.loader.start()

    def on_model_loaded(self, model, providers, aliases, voices, model_name):
        self.model = model
        combined = []
        for name in aliases + voices:
            if name not in combined:
                combined.append(name)
        if not combined:
            combined = ["expr-voice-5-m"]
        self.voice_selector.clear()
        self.voice_selector.addItems(combined)
        if "Luna" in combined:
            self.voice_selector.setCurrentText("Luna")
        self.status_label.setText(f"Model loaded: {model_name}")
        self.providers_label.setText(f"ONNX providers: {providers}")
        self.generate_button.setEnabled(True)

    def on_model_failed(self, message):
        self.status_label.setText(f"Model load failed: {message}")

    def generate_audio(self):
        if self.model is None:
            return
        text = self.text_input.toPlainText().strip()
        if not text:
            self.status_label.setText("Enter some text first.")
            return
        self.player.stop()
        self.player.setMedia(QMediaContent())
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
        except OSError:
            pass
        voice = self.voice_selector.currentText()
        speed = float(self.speed_input.value())
        self.generate_button.setEnabled(False)
        self.status_label.setText("Generating audio...")
        self.metrics_label.setText("")
        self.worker = GenerateWorker(self.model, text, voice, speed, self.output_path)
        self.worker.finished.connect(self.on_audio_ready)
        self.worker.failed.connect(self.on_audio_failed)
        self.worker.start()

    def on_audio_ready(self, path, generation_seconds, audio_seconds):
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"Audio ready: {path}")
        self.metrics_label.setText(
            f"Generation: {generation_seconds:.2f}s | Audio length: {audio_seconds:.2f}s"
        )
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.player.setVolume(100)
        self.player.play()

    def on_audio_failed(self, message):
        self.generate_button.setEnabled(True)
        self.status_label.setText(f"Audio generation failed: {message}")
        self.metrics_label.setText("")

    def closeEvent(self, event):
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
        except OSError:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KittenTTSWindow()
    window.show()
    sys.exit(app.exec_())
