## Object Detection using YOLOv8 Demo

This project demonstrates object detection using [YOLOv8](https://github.com/ultralytics/ultralytics), a state-of-the-art object detection model, [OpenCV](https://opencv.org/), [Python](https://www.python.org/), and a webcam.
The model is trained on a custom dataset of images and labels, and it can detect objects in real-time.

It is a simple and easy project cobbled together from documentation and other examples to understand how to use YOLOv8 for object detection.

It runs well on most modern laptops with a webcam. As tested, it runs well on a 2024 M4 Macbook Pro.

### Features

- Real-time object detection
- Customizable model and dataset
- Easy to use and understand

### Installation

0. Install uv and dependencies

```bash
brew install uv
```

1. Install dependencies

```bash
uv sync
```

2. Run the project

```bash
uv run main.py
```

3. Stop the project

Press `Esc` to stop the project.
