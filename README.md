# KittenTTS GUI

A friendly PyQt5 desktop interface for the fast KittenTTS neural text-to-speech models.  
Runs locally on GPU (CUDA) or CPU and speaks in real time.

![demo](screenshot\image_2026-02-25_15-44-41.png)

## Features
- GPU acceleration (CUDA) – first run is ~1–2 s, CPU fallback included
- Voice selector (Luna, Bruno, Bella, Jasper, Rosie, Hugo, Kiki, Leo)
- Speed control (0.5× – 2×)
- Auto-play preview after generation
- Single temp file – no disk clutter
- Generation time + audio length shown

## Quick Start

1. Clone or download this folder.
2. Create & activate venv:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate     # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install kittentts soundfile PyQt5
   ```
4. Run the GUI:
   ```bash
   python kitten_gui.py
   ```
5. Type text → pick voice → **Generate & Play**.  
   Metrics appear below the text box.

## GPU Support (Windows example)
If you have NVIDIA GPU + CUDA 12.x + cuDNN 9.x, the GUI already adds the required DLL paths.  
You’ll see `ONNX providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']` on launch.

## CLI fallback
`test.py` is a minimal script that writes `output.wav`; useful for batch jobs or debugging.

## Project layout
```
kitten_gui.py        – main PyQt5 window
test.py              – bare-bones CLI example
.gitignore           – ignores venv, cache, *.wav
```

## License
Same as KittenTTS (Apache-2.0).