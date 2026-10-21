import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Dự đoán rủi ro vỡ nợ thẻ tín dụng",
    page_icon="💳",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "random_forest_undersampling.joblib"
THRESHOLD = 0.4

FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

def prepare_features(data):
    """Chuẩn hóa dữ liệu theo quy tắc đã dùng trong notebook."""
    data = data.copy()
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError("Thiếu cột bắt buộc: " + ", ".join(missing))
    data = data[FEATURES].copy()

    # Giữ đúng bước gộp mã trong notebook của nhóm.
    data["EDUCATION"] = pd.to_numeric(data["EDUCATION"], errors="raise").replace({
        0: 4, 5: 4, 6: 4
    })
    data["MARRIAGE"] = pd.to_numeric(data["MARRIAGE"], errors="raise").replace({
        0: 3
    })

    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="raise")
    return data

def probability_of_default(model, X):
    classes = list(model.classes_) if hasattr(model, "classes_") else None
    probs = model.predict_proba(X)
    if classes and 1 in classes:
        return probs[:, classes.index(1)]
    return probs[:, 1]

st.title("💳 Dự đoán rủi ro vỡ nợ thẻ tín dụng")
st.caption("Ứng dụng học thuật sử dụng Random Forest – Undersampling, với ngưỡng dự đoán 0,4.")

model = load_model()
if model is None:
    st.error(
        "Chưa tìm thấy tệp mô hình `random_forest_undersampling.joblib`. "
        "Hãy xuất mô hình từ notebook theo hướng dẫn trong README.md rồi đặt tệp cạnh app.py."
    )
    st.stop()

with st.sidebar:
    st.header("Thông tin mô hình")
    st.write("**Thuật toán:** Random Forest")
    st.write("**Xử lý mất cân bằng:** Random Undersampling")
    st.write("**Ngưỡng dự đoán:** 0,4")
    st.write("**Recall lớp 1 trên Test (đã báo cáo):** khoảng 0,777")
    st.write("**ROC-AUC trên Test (đã báo cáo):** khoảng 0,766")
    st.warning(
        "Các chỉ số trên là kết quả đánh giá học thuật của nhóm, "
        "không phải cam kết hiệu năng trên dữ liệu mới."
    )

tab_single, tab_batch, tab_about = st.tabs(
    ["Dự đoán một khách hàng", "Dự đoán hàng loạt (CSV)", "Giới thiệu & lưu ý"]
)

with tab_single:
    st.subheader("Nhập thông tin khách hàng")
    st.info(
        "Nhập các trường theo đúng định nghĩa và đơn vị của bộ dữ liệu UCI. "
        "Các mã phân loại phải dùng mã số tương ứng trong dữ liệu."
    )

    with st.form("customer_form"):
        st.markdown("#### 1. Thông tin chung")
        c1, c2, c3, c4, c5 = st.columns(5)
        limit_bal = c1.number_input("Hạn mức tín dụng (LIMIT_BAL)", min_value=0.0, value=50000.0, step=10000.0)
        sex = c2.selectbox("Giới tính (SEX)", options=[1, 2], format_func=lambda x: {1:"1 - Nam", 2:"2 - Nữ"}[x])
        education = c3.selectbox("Học vấn (EDUCATION)", options=[1, 2, 3, 4], format_func=lambda x: {1:"1 - Sau đại học", 2:"2 - Đại học", 3:"3 - Trung học", 4:"4 - Khác/gộp"}[x])
        marriage = c4.selectbox("Hôn nhân (MARRIAGE)", options=[1, 2, 3], format_func=lambda x: {1:"1 - Đã kết hôn", 2:"2 - Độc thân", 3:"3 - Khác/gộp"}[x])
        age = c5.number_input("Tuổi (AGE)", min_value=18, max_value=100, value=35, step=1)

        st.markdown("#### 2. Trạng thái thanh toán")
        pay_labels = {
            -2: "-2", -1: "-1", 0: "0", 1: "1", 2: "2", 3: "3",
            4: "4", 5: "5", 6: "6", 7: "7", 8: "8"
        }
        pay_cols = st.columns(3)
        pay_values = {}
        for i, col in enumerate(PAY_COLS):
            pay_values[col] = pay_cols[i % 3].selectbox(
                f"{col} (mã trạng thái)",
                options=list(pay_labels.keys()),
                index=3 if col == "PAY_0" else 2,
                format_func=lambda x: pay_labels[x],
                key=f"single_{col}"
            )

        st.markdown("#### 3. Số dư hóa đơn (BILL_AMT1–BILL_AMT6)")
        bill_cols = st.columns(3)
        bill_values = {}
        for i, col in enumerate(BILL_COLS):
            bill_values[col] = bill_cols[i % 3].number_input(
                col, value=0.0, step=1000.0, key=f"single_{col}"
            )

        st.markdown("#### 4. Số tiền thanh toán (PAY_AMT1–PAY_AMT6)")
        amount_cols = st.columns(3)
        amount_values = {}
        for i, col in enumerate(PAY_AMT_COLS):
            amount_values[col] = amount_cols[i % 3].number_input(
                col, min_value=0.0, value=0.0, step=1000.0, key=f"single_{col}"
            )

        submitted = st.form_submit_button("Dự đoán", type="primary", use_container_width=True)

    if submitted:
        row = {
            "LIMIT_BAL": limit_bal, "SEX": sex, "EDUCATION": education,
            "MARRIAGE": marriage, "AGE": age,
            **pay_values, **bill_values, **amount_values
        }
        X_input = prepare_features(pd.DataFrame([row]))
        prob = float(probability_of_default(model, X_input)[0])
        pred = int(prob >= THRESHOLD)

        left, right = st.columns([1, 1])
        with left:
            if pred == 1:
                st.error("### Dự đoán: Có nguy cơ vỡ nợ (nhãn 1)")
            else:
                st.success("### Dự đoán: Chưa bị xếp vào nhóm vỡ nợ (nhãn 0)")
            st.metric("Xác suất dự đoán nhãn vỡ nợ", f"{prob:.1%}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.caption(f"Quy tắc: dự đoán nhãn 1 khi xác suất ≥ {THRESHOLD:.1f}.")
        with right:
            st.markdown("#### Cách diễn giải")
            st.write(
                "Đây là dự đoán thống kê của mô hình trên dữ liệu đầu vào, "
                "không phải kết luận chắc chắn rằng khách hàng sẽ hoặc sẽ không vỡ nợ."
            )
            st.write("Nên sử dụng kết quả để minh họa hoặc hỗ trợ sàng lọc, không dùng đơn lẻ để ra quyết định tín dụng.")

with tab_batch:
    st.subheader("Dự đoán nhiều khách hàng từ CSV")
    st.write(
        "Tải tệp CSV có đủ 23 cột đầu vào theo tên cột trong bộ UCI. "
        "Không cần cột `ID` hoặc `default payment next month`."
    )
    sample = pd.DataFrame(columns=FEATURES)
    st.download_button(
        "Tải CSV mẫu chỉ chứa tiêu đề cột",
        data=sample.to_csv(index=False).encode("utf-8-sig"),
        file_name="mau_du_lieu_dau_vao.csv",
        mime="text/csv"
    )
    uploaded = st.file_uploader("Chọn tệp CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded)
            X_batch = prepare_features(raw)
            if st.button("Chạy dự đoán hàng loạt", type="primary"):
                probs = probability_of_default(model, X_batch)
                preds = (probs >= THRESHOLD).astype(int)
                result = raw.copy()
                result["xac_suat_vo_no"] = probs
                result["du_doan_vo_no"] = preds
                result["ket_qua"] = np.where(
                    preds == 1, "Có nguy cơ vỡ nợ (1)", "Dự đoán nhãn 0"
                )
                st.success(f"Đã dự đoán {len(result):,} dòng.")
                st.dataframe(result.head(100), use_container_width=True)
                st.download_button(
                    "Tải kết quả dự đoán (CSV)",
                    data=result.to_csv(index=False).encode("utf-8-sig"),
                    file_name="ket_qua_du_doan.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Không thể xử lý tệp: {e}")

with tab_about:
    st.subheader("Phạm vi sử dụng")
    st.markdown(
        """
        - Mô hình được huấn luyện từ bộ dữ liệu UCI Default of Credit Card Clients.
        - Ngưỡng 0,4 được nhóm lựa chọn trên tập Validation.
        - Tệp CSV cần có đủ 23 đặc trưng và dùng cùng định nghĩa/mã hóa như dữ liệu huấn luyện.
        - Ứng dụng không tự xác minh danh tính, tính hợp pháp hoặc độ chính xác của dữ liệu khách hàng.
        - Không nhập thông tin nhận dạng trực tiếp như họ tên, số thẻ, số căn cước hoặc thông tin bí mật.
        - Đây là nguyên mẫu phục vụ học tập, không phải hệ thống chấm điểm tín dụng đã được thẩm định để triển khai thương mại.
        """
    )
    st.markdown("#### Các cột đầu vào bắt buộc")
    st.code(", ".join(FEATURES))
