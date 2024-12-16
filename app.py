from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Khởi tạo Flask app
app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Load mô hình đã huấn luyện
model_path = "model/trash_model.h5"
model = load_model(model_path)

# Tên các loại rác
class_names = [
    "Cardboard",
    "Glass",
    "Metal",
    "Paper",
    "Plastic",
    "Trash",
]  # Cập nhật theo mô hình


# Kiểm tra định dạng file
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


# Xử lý ảnh từ luồng tải lên
def preprocess_image(file_stream):
    # Mở ảnh từ luồng
    img = Image.open(file_stream).convert("RGB")
    # Thay đổi kích thước về (384, 384)
    img = img.resize((384, 384))
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
        return "Không tìm thấy file", 400

    file = request.files["file"]
    if file.filename == "":
        return "Không có file nào được chọn", 400

    if file and allowed_file(file.filename):
        try:
            # Xử lý ảnh trực tiếp từ luồng
            img_array = preprocess_image(file.stream)

            # Dự đoán
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]

            return f"Kết quả dự đoán: {predicted_class}"
        except Exception as e:
            return f"Lỗi trong quá trình xử lý: {str(e)}", 500

    return "File không hợp lệ", 400


if __name__ == "__main__":
    app.run(debug=True)
