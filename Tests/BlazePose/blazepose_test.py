"""
BlazePose Fullbody Python inference test.

This test downloads the pose_landmark_heavy.onnx model and runs inference
using onnxruntime, verifying the output shapes and basic properties.

Python reference: blazepose-fullbody/blazepose-fullbody.py
Model: pose_landmark_heavy (NHWC input [1,256,256,3])

Requirements:
    pip install numpy onnxruntime requests pytest
"""

import os
import sys
import urllib.request
import numpy as np
import pytest

MODEL_URL = "https://storage.googleapis.com/ailia-models/blazepose-fullbody/pose_landmark_heavy.onnx"
DETECTION_MODEL_URL = "https://storage.googleapis.com/ailia-models/blazepose-fullbody/pose_detection.onnx"
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


def download_model(url, cache_dir=CACHE_DIR):
    """Download model if not cached."""
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    filepath = os.path.join(cache_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    return filepath


def sigmoid(x):
    """Python reference sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


# =======================================================
# 1. Sigmoid tests (match C# Sigmoid)
# =======================================================
class TestSigmoid:
    def test_zero(self):
        assert abs(sigmoid(0) - 0.5) < 1e-5

    def test_large_positive(self):
        assert sigmoid(10) > 0.999

    def test_large_negative(self):
        assert sigmoid(-10) < 0.001

    def test_symmetry(self):
        for x in [0.5, 1.0, 2.5, 7.0]:
            assert abs(sigmoid(x) + sigmoid(-x) - 1.0) < 1e-5

    def test_known_values(self):
        """Same values as C# test."""
        assert abs(sigmoid(-5) - 0.00669285) < 1e-4
        assert abs(sigmoid(-1) - 0.26894142) < 1e-4
        assert abs(sigmoid(1) - 0.73105858) < 1e-4
        assert abs(sigmoid(5) - 0.99330715) < 1e-4


# =======================================================
# 2. IoU / Jaccard overlap tests (match C# Box.GetJaccardOverlap)
# =======================================================
def jaccard_overlap(box1, box2):
    """Python IoU implementation matching C# Box.GetJaccardOverlap."""
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    intersection = x_overlap * y_overlap
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


class TestJaccardOverlap:
    def test_identical(self):
        box = [0, 0, 1, 1]
        assert abs(jaccard_overlap(box, box) - 1.0) < 1e-5

    def test_no_overlap(self):
        assert abs(jaccard_overlap([0, 0, 1, 1], [2, 2, 3, 3])) < 1e-5

    def test_partial_overlap(self):
        iou = jaccard_overlap([0, 0, 2, 2], [1, 1, 3, 3])
        assert abs(iou - 1.0 / 7.0) < 1e-5


# =======================================================
# 3. Landmark decoding tests (match C# DecodeAndProcessLandmarks)
# =======================================================
def decode_landmarks(output_buffer, resolution=256):
    """Python reference landmark decoding matching C# DecodeAndProcessLandmarks."""
    landmarks = []
    for i in range(33):
        x = output_buffer[i * 5] / resolution
        y = output_buffer[i * 5 + 1] / resolution
        z = output_buffer[i * 5 + 2] / resolution
        visibility = output_buffer[i * 5 + 3]
        presence = output_buffer[i * 5 + 4]
        confidence = sigmoid(min(visibility, presence))
        landmarks.append({"x": x, "y": y, "z": z, "confidence": confidence})
    return landmarks


class TestLandmarkDecoding:
    def test_center_landmarks(self):
        buf = np.zeros(195, dtype=np.float32)
        for i in range(33):
            buf[i * 5 + 0] = 128  # x
            buf[i * 5 + 1] = 128  # y
            buf[i * 5 + 2] = 0    # z
            buf[i * 5 + 3] = 5    # visibility
            buf[i * 5 + 4] = 5    # presence
        lms = decode_landmarks(buf)
        assert len(lms) == 33
        for lm in lms:
            assert abs(lm["x"] - 0.5) < 1e-5
            assert abs(lm["y"] - 0.5) < 1e-5
            assert lm["confidence"] > 0.9

    def test_confidence_uses_min(self):
        buf = np.zeros(195, dtype=np.float32)
        buf[3] = 10.0   # visibility high
        buf[4] = -2.0   # presence low
        lms = decode_landmarks(buf)
        expected = sigmoid(-2.0)
        assert abs(lms[0]["confidence"] - expected) < 1e-4


# =======================================================
# 4. Model inference tests (require onnxruntime)
# =======================================================
class TestEstimationModelInference:
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import onnxruntime
            self.ort = onnxruntime
        except ImportError:
            pytest.skip("onnxruntime not installed")

    def test_estimation_model_output_shapes(self):
        """Verify pose_landmark_heavy model output shapes."""
        model_path = download_model(MODEL_URL)
        session = self.ort.InferenceSession(model_path)

        # NHWC input: [1, 256, 256, 3]
        input_data = np.random.rand(1, 256, 256, 3).astype(np.float32)
        input_name = session.get_inputs()[0].name
        assert input_name == "input_1"

        outputs = session.run(None, {input_name: input_data})

        # 5 outputs: Identity[1,195], Identity_1[1,1], Identity_2[1,128,128,1],
        #            Identity_3[1,64,64,39], Identity_4[1,117]
        assert len(outputs) == 5
        assert outputs[0].shape == (1, 195), f"Identity shape: {outputs[0].shape}"
        assert outputs[1].shape == (1, 1), f"Identity_1 shape: {outputs[1].shape}"
        assert outputs[2].shape == (1, 128, 128, 1), f"Identity_2 shape: {outputs[2].shape}"
        assert outputs[3].shape == (1, 64, 64, 39), f"Identity_3 shape: {outputs[3].shape}"
        assert outputs[4].shape == (1, 117), f"Identity_4 shape: {outputs[4].shape}"

    def test_estimation_model_landmark_range(self):
        """Verify landmarks are in reasonable range for a valid input."""
        model_path = download_model(MODEL_URL)
        session = self.ort.InferenceSession(model_path)

        # Create a simple test image (gray)
        input_data = np.full((1, 256, 256, 3), 0.5, dtype=np.float32)
        outputs = session.run(None, {"input_1": input_data})

        landmarks = outputs[0][0]  # [195]
        # 33 landmarks * 5 = 165, remaining 30 are auxiliary
        for i in range(33):
            x = landmarks[i * 5]
            y = landmarks[i * 5 + 1]
            # Coordinates should be within model resolution range
            assert -256 < x < 512, f"Landmark {i} x={x} out of range"
            assert -256 < y < 512, f"Landmark {i} y={y} out of range"

    def test_estimation_model_predict_api(self):
        """Test using predict-style API (single run, all outputs)."""
        model_path = download_model(MODEL_URL)
        session = self.ort.InferenceSession(model_path)

        input_data = np.random.rand(1, 256, 256, 3).astype(np.float32) / 255.0
        outputs = session.run(None, {"input_1": input_data})

        # Score output (Identity_1) should be a single value
        score = sigmoid(outputs[1][0, 0])
        assert 0 <= score <= 1, f"Score {score} out of range"


class TestDetectionModelInference:
    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            import onnxruntime
            self.ort = onnxruntime
        except ImportError:
            pytest.skip("onnxruntime not installed")

    def test_detection_model_output_shapes(self):
        """Verify pose_detection model output shapes."""
        model_path = download_model(DETECTION_MODEL_URL)
        session = self.ort.InferenceSession(model_path)

        # NHWC input: [1, 224, 224, 3]
        input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        input_name = session.get_inputs()[0].name
        assert input_name == "input_1"

        outputs = session.run(None, {input_name: input_data})

        # 2 outputs: Identity[1,2254,12], Identity_1[1,2254,1]
        assert len(outputs) == 2
        assert outputs[0].shape == (1, 2254, 12), f"Identity shape: {outputs[0].shape}"
        assert outputs[1].shape == (1, 2254, 1), f"Identity_1 shape: {outputs[1].shape}"


# =======================================================
# 5. Affine transform test (match C# GetResult)
# =======================================================
def affine_transform_landmark(px, py, affine_xc, affine_yc, affine_scale, affine_angle):
    """Python reference affine transform matching C# GetResult."""
    cs = np.cos(-affine_angle)
    ss = np.sin(-affine_angle)
    x = ((px - 0.5) * cs + (py - 0.5) * ss) * affine_scale + affine_xc
    y = ((px - 0.5) * (-ss) + (py - 0.5) * cs) * affine_scale + affine_yc
    return x, y


class TestAffineTransform:
    def test_identity_transform(self):
        x, y = affine_transform_landmark(0.5, 0.5, 0.5, 0.5, 1.0, 0.0)
        assert abs(x - 0.5) < 1e-5
        assert abs(y - 0.5) < 1e-5

    def test_rotation_90(self):
        angle = np.pi / 2
        x, y = affine_transform_landmark(0.75, 0.5, 0.5, 0.5, 1.0, angle)
        cs = np.cos(-angle)
        ss = np.sin(-angle)
        expected_x = (0.25 * cs + 0.0 * ss) * 1.0 + 0.5
        expected_y = (0.25 * (-ss) + 0.0 * cs) * 1.0 + 0.5
        assert abs(x - expected_x) < 1e-5
        assert abs(y - expected_y) < 1e-5

    def test_scale(self):
        x, y = affine_transform_landmark(0.75, 0.5, 0.5, 0.5, 2.0, 0.0)
        # (0.25 * 2 + 0.5, 0 * 2 + 0.5) = (1.0, 0.5)
        assert abs(x - 1.0) < 1e-5
        assert abs(y - 0.5) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
