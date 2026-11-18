# ML_ASL_Glove

An American Sign Language (ASL) to English translation project combining a **computer vision + color-mask TensorFlow Lite pipeline** with an earlier **Arduino flex-sensor hardware prototype**.

## Project Overview & Pivot

This repository captures two iterations of the project:

1. **Color-Mask Vision Pipeline (`Model_pipeline.py`)** — The active software track. Uses a high-visibility color-coded glove (distinct colors on the fingers and wrist) paired with a color mask so hand and finger regions are cleanly segmented and parsed by `sign_language_glove.tflite`, alongside hand detection, landmark, depth, and facial emotion TFLite models.
2. **Arduino Sensor Glove (`sketch_mar17a.ino`)** — The initial hardware prototype built on an **Arduino Nano 33 BLE Sense Rev2** (`Arduino_BMI270_BMM150`), an external **Adafruit MPU6050** IMU, and four analog flex sensors (`A7`, `A1`, `A2`, `A3`). After physical wear and hardware breakdown on the flex-sensor glove, the project pivoted to the color-mask CV approach for better durability and cleaner data capture.

## Models Used (`Model_pipeline.py`)

`Model_pipeline.py` runs inference via `ai_edge_litert.interpreter` using six `.tflite` models:
* `detector.tflite` — Hand detection
* `landmarks.tflite` — Hand landmark extraction
* `asl.tflite` — ASL classification
* `sign_language_glove.tflite` — Color-masked glove gesture classification
* `midas.tflite` — Monocular depth estimation
* `face_emotion.tflite` — Facial expression context

## Setup & Usage

### 1. Python Vision Pipeline
Install the required Python packages (`opencv-python`, `numpy`, `pillow`, `ai-edge-litert`, `pyttsx3`) and run:

```bash
python Model_pipeline.py
```

*(Note: `Model_pipeline.py` currently uses `winsound` for audio alerts on Windows.)*

### 2. Arduino Hardware Prototype (Archived)
Open `sketch_mar17a.ino` in the Arduino IDE, install `Arduino_BMI270_BMM150` and `Adafruit_MPU6050`, select **Arduino Nano 33 BLE**, and upload.

## License

MIT more robust changes to the structure of the glove to prevent such failure.
