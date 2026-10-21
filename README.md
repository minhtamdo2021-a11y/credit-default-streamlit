
# Credit Card Default Prediction

## 1. Giới thiệu dự án

Dự án xây dựng ứng dụng học thuật dự đoán rủi ro vỡ nợ thẻ tín dụng dựa trên thông tin hạn mức tín dụng, nhân khẩu học, lịch sử thanh toán, hóa đơn và số tiền thanh toán của khách hàng.

Ứng dụng được phát triển bằng Streamlit, cho phép nhập dữ liệu của một khách hàng hoặc tải dữ liệu CSV để thử nghiệm dự đoán hàng loạt.

## 2. Dữ liệu

Dự án sử dụng bộ dữ liệu Default of Credit Card Clients từ UCI Machine Learning Repository.

Dữ liệu gồm 30.000 quan sát và 25 cột trong file gốc, bao gồm mã khách hàng, các biến đầu vào và biến mục tiêu `default payment next month`.

- Nhãn 0: Không vỡ nợ trong kỳ tiếp theo.
- Nhãn 1: Vỡ nợ trong kỳ tiếp theo.

## 3. Phương pháp

Mô hình được lựa chọn là Random Forest kết hợp Random Undersampling để xử lý tình trạng mất cân bằng giữa hai lớp.

Ngưỡng dự đoán được thiết lập ở mức 0,4:

- Xác suất dự đoán từ 0,4 trở lên: nhãn 1.
- Xác suất dưới 0,4: nhãn 0.

Kết quả mang tính học thuật, không phải kết luận chắc chắn về khả năng trả nợ của khách hàng.

## 4. Công nghệ sử dụng

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit

## 5. Cấu trúc dự án

```text
credit-default-streamlit/
├── README.md
├── app.py
├── requirements.txt
├── random_forest_undersampling.joblib
├── notebooks/
├── data/
└── reports/
    └── figures/
```

## 6. Cài đặt và chạy ứng dụng

Cài Python tương thích với các thư viện trong `requirements.txt`.

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Khởi chạy ứng dụng:

```bash
streamlit run app.py
```

## 7. Hướng dẫn sử dụng

1. Mở ứng dụng Streamlit.
2. Nhập thông tin khách hàng hoặc sử dụng chức năng dự đoán hàng loạt nếu có file CSV phù hợp.
3. Nhấn nút dự đoán.
4. Xem nhãn dự đoán và xác suất do mô hình trả về.

## 8. Giới hạn

Đây là sản phẩm phục vụ học tập và thử nghiệm mô hình học máy. Kết quả phụ thuộc vào dữ liệu đầu vào và mô hình đã huấn luyện, không được sử dụng độc lập để quyết định cấp tín dụng thực tế.
