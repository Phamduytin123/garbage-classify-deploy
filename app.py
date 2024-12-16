from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
from keras.utils import custom_object_scope


# Khởi tạo Flask app
app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

from keras.layers import Layer


class TensorFlowOpLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Custom build logic here
        pass

    def call(self, inputs, **kwargs):
        # Custom call logic here (e.g., apply TensorFlow op)
        return inputs

    @classmethod
    def from_config(cls, config):
        # Override to handle custom config
        # Ignore or process unrecognized arguments
        config.pop("node_def", None)  # Remove 'node_def' from config
        config.pop("constants", None)  # Remove 'constants' from config
        return cls(**config)


# Define your models dictionary
models = {
    "ResNet50": "model/resnet50.h5",
    "MobileNet": "model/densenet.h5",
    "EfficentNet": "model/efficientnet.keras",
    "DenseNet": "model/densenet.h5",
}

# Use custom_object_scope to load the model with the custom layer
with custom_object_scope({"TensorFlowOpLayer": TensorFlowOpLayer}):
    loaded_models = {name: load_model(path) for name, path in models.items()}

# Tên các loại rác
class_names = [
    "Battery",
    "Biological",
    "Brown-glass",
    "Cardboard",
    "Clothes",
    "Green-glass",
    "Metal",
    "Paper",
    "Plastic",
    "Shoes",
    "Trash",
    "White-glass",
]  # Cập nhật theo mô hình


# Kiểm tra định dạng file
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Xử lý ảnh từ luồng tải lên
def preprocess_image(file_stream, model_name):
    # Mở ảnh từ luồng
    img = Image.open(file_stream).convert("RGB")

    # Chuyển kích thước ảnh phù hợp với từng mô hình
    if model_name == "ResNet50":
        img = img.resize((384, 384))
        # ResNet50 yêu cầu kích thước 224x224
    else:
        if model_name == "MobileNet":
            # img = img.resize((128, 128))
            img = img.resize((224, 224))
        else:
            img = img.resize((224, 224))

    # Chuyển thành mảng numpy
    img_array = np.array(img)
    # Chuẩn hóa giá trị pixel
    img_array = img_array / 255.0
    # Thêm một chiều batch
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Trang chính
@app.route("/")
def index():
    return render_template("index.html")


# Xử lý upload và dự đoán
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Không có file nào được chọn"}), 400

    if file and allowed_file(file.filename):
        try:
            # Xử lý ảnh trực tiếp từ luồng
            results = []
            for model_name, model in loaded_models.items():
                # Xử lý ảnh cho mỗi mô hình
                img_array = preprocess_image(file.stream, model_name)

                # Dự đoán với mô hình
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions)]
                results.append({"model": model_name, "prediction": predicted_class})

            # Trả về kết quả dưới dạng JSON
            return jsonify({"predictions": results})
        except Exception as e:
            return jsonify({"error": f"Lỗi trong quá trình xử lý: {str(e)}"}), 500

    return jsonify({"error": "File không hợp lệ"}), 400


if __name__ == "__main__":
    app.run(debug=True)
