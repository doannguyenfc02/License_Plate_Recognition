from flask import Flask, request, jsonify
from PIL import Image
import base64
from io import BytesIO
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper

app = Flask(__name__)

yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

def process_image(img_path):
    img = cv2.imread(img_path)
    plates = yolo_LP_detect(img, size=640)

    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    license_plate = None
    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate,img)
        if lp != "unknown":
            license_plate = lp
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0])  # xmin
            y = int(plate[1])  # ymin
            w = int(plate[2] - plate[0])  # xmax - xmin
            h = int(plate[3] - plate[1])  # ymax - ymin
            crop_img = img[y:y+h, x:x+w]
            lp = ""
            for cc in range(0, 2):
                for ct in range(0, 2):
                    lp = helper.read_plate(
                        yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        license_plate = lp
                        list_read_plates.add(lp)
                        flag = 1
                        break
                if flag == 1:
                    break

    return license_plate, list_read_plates

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Nhận dữ liệu JSON từ yêu cầu
        data = request.json

        # Kiểm tra xem trường 'file' có tồn tại không
        if 'file' not in data:
            return jsonify({'error': 'Missing "file" field'}), 400

        # Giải mã mã base64
        base64_encoded = data['file']
        file_content = base64.b64decode(base64_encoded)

        # Chuyển đổi nội dung thành hình ảnh
        image = Image.open(BytesIO(file_content))

        # Lưu hình ảnh thành tệp
        image.save('static/uploaded_image.jpg')

        # Xử lý ảnh và nhận diện
        license_plate, list_read_plates = process_image('static/uploaded_image.jpg')

        # Trả về JSON response
        response_data = {
            'license_plate': license_plate,
            'list_read_plates': list(list_read_plates)
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
