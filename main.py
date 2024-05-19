import cv2
import json
import numpy as np
import yaml


def load_config(config_file):
    """Load configuration from a YAML file."""
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    class Config:
        def __init__(self, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    value = Config(value)
                setattr(self, key, value)

    return Config(config_data)


def load_model(model_path):
    """Load CascadeClassifier model for face detection."""
    model = cv2.CascadeClassifier(model_path)
    if model.empty():
        raise IOError("Error: Could not load face cascade classifier.")
    print("Face cascade classifier loaded successfully.")
    return model


def convert_to_builtin_type(obj):
    """Convert numpy arrays and types to Python built-in types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_json(face_data, output_path):
    """Write face data to a JSON file."""
    with open(output_path, "w") as json_file:
        json.dump(face_data, json_file, default=convert_to_builtin_type)


def process_video(config):
    """Process the video to detect faces and log the face data."""

    face_cascade = load_model(config.paths.inputs.model_path)

    cap = cv2.VideoCapture(config.paths.inputs.video_path)
    if not cap.isOpened():
        raise IOError("Error: Unable to open the video file.")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to write the processed frames to a new video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(config.paths.outputs.video_path, fourcc, fps, (width, height))

    # Dictionary to store face data for each frame
    face_data = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.model.scale_factor,
            minNeighbors=config.model.min_neighbors,
            minSize=(config.model.min_size.w, config.model.min_size.h),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        frame_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            if w <= config.model.max_size.w and h <= config.model.max_size.h:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"({x}, {y})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                frame_faces.append({f"face{i+1}": {"x": x, "y": y, "w": w, "h": h}})

        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.putText(
            frame,
            f"Frame: {frame_num}",
            (width - 150, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 225),
            1,
        )

        face_data[f"frame{frame_num}"] = {
            "faces": frame_faces,
            "time": cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,  # Convert time to seconds
        }

        if config.process.if_save_video:
            out.write(frame)

    cap.release()
    out.release()
    if config.process.is_save_json:
        write_json(face_data, config.paths.outputs.json_path)


if __name__ == "__main__":
    config = load_config("conf.yaml")
    process_video(config)
    print("--- DONE ---")
