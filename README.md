--- DANH SÁCH THÀNH VIÊN ---
Nguyễn Thái Học - 22520488
Nguyễn Duy Khang - 22520619
Nguyễn Trọng Nhân - 22521005

--- LINK DATASET ---
Cho SSD/ADSSD: https://drive.google.com/file/d/1WNy9IapqRqmSOTDVBzBY9YpHS_DhnjKF/view?usp=sharing
Cho YOLOv8: https://www.kaggle.com/datasets/lakshaytyagi01/fruit-detection?fbclid=IwZXh0bgNhZW0CMTEAAR1mCbBNO2o2oadRmTn61xJ9mq0Uw9jXYddV585-2Zgxa2sgYYssR85tcrE_aem_lfpSzIsset4hBnfJnUAYCQ

--- LINK SOURCE CODE---
https://github.com/HocNguyen10112004/Fruit_Detection.git

--- HƯỚNG DẪN CÀI ĐẶT MÔI TRƯỜNG ẢO ---

*** YÊU CẦU ***
Anaconda Prompt (miniconda3)
nvcc
nvidia driver

*** CÀI ĐẶT ***
# Chạy trong miniconda3 promt
git clone https://github.com/HocNguyen10112004/Fruit_Detection/tree/master/voc
cd cs406
conda env create --file environment.yml

--- TRAINING ---

*** YÊU CẦU ***
Giải nén thư mục train vào thư mục ssd / adssd

*** TRAIN ***
	+ Mở miniconda3 tại thư mục ssd hoặc adssd
	+ conda activate pytorch_env
	+ python -m tools.train
--- EVALUATION ---

*** YÊU CẦU ***
Giải nén thư mục test vào thư mục ssd / adssd

*** EVALUATION ***
	+ Mở miniconda3 tại thư mục ssd hoặc adssd
	+ conda activate pytorch_env
	+ python -m tools.infer --evaluate True --infer_samples False

--- DEMO ---

*** YÊU CẦU ***
Có sẵn các model trong thư mục voc (nếu clone từ github sẽ có sẵn)
Link thư mục chứa các model: https://github.com/HocNguyen10112004/Fruit_Detection.git/tree/master/voc (tải về đưa vào thư mục voc)
*** DEMO ***
	+ Mở miniconda3 tại thư mục làm việc chính (source / cs406)
	+ conda activate pytorch_env
	+ streamlit run demo.py

--- YOLOV8 ---
Chạy file jupyter notebook yolov8-fruit-detection.ipynb trên kaggle.
