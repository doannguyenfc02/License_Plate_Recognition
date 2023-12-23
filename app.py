from flask import Flask, render_template, request
from PIL import Image
import cv2
import torch
import function.utils_rotate as utils_rotate
import function.helper as helper
import os

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
            cv2.putText(img, lp, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0]) # xmin
            y = int(plate[1]) # ymin
            w = int(plate[2] - plate[0]) # xmax - xmin
            h = int(plate[3] - plate[1]) # ymax - ymin  
            crop_img = img[y:y+h, x:x+w]
            cv2.rectangle(img, (int(plate[0]),int(plate[1])), (int(plate[2]),int(plate[3])), color = (0,0,225), thickness = 2)
            cv2.imwrite("static/crop.jpg", crop_img)
            rc_image = cv2.imread("static/crop.jpg")
            lp = ""
            for cc in range(0,2):
                for ct in range(0,2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        license_plate = lp
                        list_read_plates.add(lp)
                        cv2.putText(img, lp, (int(plate[0]), int(plate[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        flag = 1
                        break
                if flag == 1:
                    break

    cv2.imwrite('static/result.jpg', img)
    return license_plate

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='Không có phần file nào được chọn')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='Không có file nào được chọn')

        if file:
            # Đảm bảo thư mục 'uploads' tồn tại
            if not os.path.exists('static'):
                os.makedirs('static')

            # Lưu file được tải lên
            file_path = os.path.join('static', file.filename)
            file.save(file_path)

            # Xử lý ảnh và nhận diện biển số
            license_plate = process_image(file_path)

            result_path = 'static/result.jpg'

            return render_template('index.html', result=result_path, license_plate=license_plate)

    return render_template('index.html', error=None, result=None, license_plate=None)

if __name__ == '__main__':
    app.run(debug=True)
