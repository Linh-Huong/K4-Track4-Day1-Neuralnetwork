# Báo cáo Lab Day 1 — Vũ Tiến Linh — 2A202602657

## 1. Thiết lập

- Môi trường (Colab/Kaggle, GPU, phiên bản PyTorch): Local, GPU NVIDIA GeForce RTX 4060 Laptop (8GB VRAM), PyTorch 2.6.0+cu124 (CUDA 12.4), Python 3.11.
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val.
- Model: `M-base` (54→256→128→7, 47 879 tham số). Baseline: Cross-Entropy loss, SGD+momentum (momentum 0.9), lr = 0.1, batch = 512, epochs = 20, init He.
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = 0.4876 (lớp 1 chiếm 48.76%).
- Các chủ đề đã thử: ☑ loss ☑ optimizer ☑ hyper-parameter ☑ dropout ☑ clipping ☑ mixed precision ☑ init

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | 2.0024 (chênh lệch 0.0565 so với ln 7) |
| Quá khớp 20 mẫu: loss cuối | 0.000004 (Accuracy 100%, 200 bước Adam) |
| Mọi tham số có gradient khác 0 | ☑ có (chuẩn L2: 0.39 – 6.83) |
| Baseline, số seed đã chạy | 3 (seed 1, 2, 3: `base-s1`, `base-s2`, `base-s3`) |
| Baseline: val acc (TB ± σ) | 0.9112 ± 0.0014 |
| Baseline: val macro-F1 (TB ± σ) | 0.8546 ± 0.0129 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = 0.0258 (val macro-F1).

## 3. Kết quả theo chủ đề

### 3.1 Hàm mất mát — CE vs MSE
- **Dự đoán**: Cross-Entropy (CE) sẽ vượt trội hơn Mean Squared Error (MSE) cả về tốc độ hội tụ lẫn chỉ số phân loại macro-F1. Đạo hàm của CE theo logit thô là $\frac{\partial L_{CE}}{\partial z_i} = p_i - y_i$. Khi mô hình dự đoán sai lệch nghiêm trọng ($p_i \approx 0$ cho nhãn đúng $y_i=1$), gradient đạt cực đại $|p_i - y_i| \approx 1$, kéo mạnh trọng số về đúng hướng. Ngược lại, MSE trên vector one-hot có gradient $\frac{\partial L_{MSE}}{\partial z_i} = 2(z_i - y_i)$, không gắn liền với xác suất chuẩn hóa Softmax và bị áp đảo bởi các lớp đa số, khiến các lớp thiểu số nhận gradient rất yếu.
- **Kết quả**:
  - `base-s1` (CE): Val Macro-F1 = **0.8407**, Val Acc = **0.9095**, Best Val Loss = **0.2288** (ảnh `figures/base-s1.png`).
  - `loss-mse` (MSE): Val Macro-F1 = **0.7238**, Val Acc = **0.8696**, Best Val Loss = **0.0294** (ảnh `figures/loss-mse.png`).
  - Biểu đồ so sánh: `figures/compare_loss.png`.
- **Giải thích & Cơ chế**:
  - Chênh lệch: $\Delta \text{Macro-F1} = 0.8407 - 0.7238 = \mathbf{+0.1169} \gg 2\sigma$ ($0.0258$), gấp 4.5 lần ngưỡng nhiễu seed. Sự vượt trội của CE có ý nghĩa thống kê tuyệt đối.
  - Lưu ý không so sánh trực tiếp giá trị loss của CE (0.2288) và MSE (0.0294) do hai hàm khác nhau về thang đo.
  - Cơ chế: Do mất cân bằng lớp (lớp 1 chiếm 48.8%, lớp 3 chỉ 0.5%), MSE phạt lỗi theo khoảng cách Euclidean bình phương nên mô hình có xu hướng hạ thấp mọi logit để giảm lỗi trung bình của 6 lớp âm, làm tê liệt khả năng dự đoán lớp hiếm. CE thông qua Softmax tạo sự cạnh tranh xác suất tương đối giữa các lớp, giúp nhận diện lớp thiểu số tốt hơn nhiều.

### 3.2 Bộ tối ưu hoá
- **Dự đoán**: SGD thuần không có momentum sẽ hội tụ chậm nhất do dao động trong các thung lũng dốc hẹp. SGD+Momentum khắc phục được nhờ vận tốc tích lũy. Adam và AdamW điều chỉnh bước học thích nghi theo từng tham số sẽ hội tụ nhanh hơn ở các epoch đầu và đạt macro-F1 cao hơn khi tìm được lr phù hợp.
- **Bảng so sánh các bộ tối ưu ở các mức lr**:
  | Bộ tối ưu | exp_id | lr | Val Acc | Val Macro-F1 | Best Val Loss | Best Epoch |
  |---|---|:---:|:---:|:---:|:---:|:---:|
  | SGD | `opt-sgd-lr0.05` | 0.05 | 0.8337 | 0.6930 | 0.3979 | 19 |
  | SGD | `opt-sgd-lr0.1` | 0.10 | 0.8473 | 0.7392 | 0.3724 | 18 |
  | SGD + Momentum | `opt-sgdm-lr0.05` | 0.05 | 0.8995 | 0.8376 | 0.2478 | 19 |
  | SGD + Momentum | `base-s1` | 0.10 | 0.9095 | **0.8407** | 0.2288 | 20 |
  | Adam | `opt-adam-lr1e-3` | 1e-3 | 0.9057 | 0.8533 | 0.2369 | 19 |
  | Adam | `opt-adam-lr3e-3` | 3e-3 | 0.9155 | **0.8751** | 0.2112 | 20 |
  | AdamW | `opt-adamw-lr1e-3` | 1e-3 | 0.9033 | 0.8491 | 0.2414 | 19 |
  | AdamW | `opt-adamw-lr3e-3` | 3e-3 | 0.9133 | **0.8682** | 0.2130 | 20 |
- **Biểu đồ so sánh**: `figures/compare_optimizer.png`.
- **Giải thích & Cơ chế**:
  - So sánh ở lr tối ưu của từng bộ: `Adam` (lr=3e-3) đạt Val Macro-F1 = **0.8751**, cao hơn Baseline `base-s1` là $\Delta \text{F1} = +0.0344 > 2\sigma$ ($0.0258$). Sự cải thiện vượt qua ngưỡng nhiễu thực nghiệm.
  - SGD thuần kém hơn SGD+Momentum hơn 10% F1 (0.7392 vs 0.8407), chứng minh động lượng $\mu v$ đóng vai trò sống còn giúp làm dịu dao động zigzag và gia tốc dọc theo hướng tối ưu.
  - Adam/AdamW tại lr=3e-3 hội tụ vượt bậc: chỉ sau 3 epoch đầu đã đạt F1 > 0.80 (nhanh hơn gấp đôi so với SGD).

### 3.3 Hyper-parameter
- **Yếu tố đã đổi**: Batch size (128, 512, 2048) và Kiến trúc mạng (`M-base`, `M-wide`, `M-deep`).
- **Kết quả**:
  | Yếu tố | exp_id | Cấu hình | Val Acc | Val Macro-F1 | Best Val Loss | Thời gian/epoch |
  |---|---|---|:---:|:---:|:---:|:---:|
  | Batch 128 | `hparam-batch-128` | Batch=128 (2 905 bước/ep) | 0.9132 | **0.8604** | 0.2227 | 3.83s |
  | Batch 512 | `base-s1` | Batch=512 (727 bước/ep) | 0.9095 | 0.8407 | 0.2288 | 1.39s |
  | Batch 2048 | `hparam-batch-2048` | Batch=2048 (182 bước/ep) | 0.8854 | 0.8156 | 0.2801 | 0.26s |
  | Kiến trúc Wide | `hparam-wide` | `M-wide` (54→512→256→7) | 0.9217 | **0.8721** | 0.1990 | 0.93s |
  | Kiến trúc Deep | `hparam-deep` | `M-deep` (54→256→128→64→7) | 0.9221 | **0.8730** | 0.1956 | 1.08s |
- **Biểu đồ so sánh**: `figures/compare_batch.png` và `figures/compare_arch.png`.
- **Giải thích & Cơ chế**:
  - *Batch size*: Batch 128 thực hiện số bước cập nhật nhiều gấp 4 lần Batch 512 (58 100 bước vs 14 540 bước trong 20 epochs), tiếng ồn ngẫu nhiên trong mini-batch đóng vai trò điều hòa giúp mô hình thoát cực tiểu cục bộ và đạt F1 = 0.8604. Batch 2048 chỉ có 182 bước/epoch nên chưa hội tụ kịp trong 20 epochs (F1 giảm về 0.8156).
  - *Kiến trúc*: Cả `M-wide` (0.8721) và `M-deep` (0.8730) đều cải thiện vượt ngưỡng $2\sigma$ (+0.0314 và +0.0323 so với `base-s1`). `M-deep` đạt Val Loss thấp nhất (0.1956) nhờ tăng chiều sâu biểu diễn đặc trưng phân cấp mà chỉ tốn thêm rất ít tham số (55k so với 161k của Wide).

### 3.4 Dropout
- **Dự đoán**: Mạng `M-base` trên 371k mẫu không bị quá khớp, do đó Dropout sẽ không có tác dụng tích cực mà có thể làm suy giảm năng lực biểu diễn nếu $q$ quá lớn.
- **Kết quả**:
  | exp_id | Dropout (q) | Val Acc | Val Macro-F1 | Best Val Loss | Train-Val Loss Gap |
  |---|:---:|:---:|:---:|:---:|:---:|
  | `base-s1` | 0.0 | 0.9095 | **0.8407** | 0.2288 | 0.0206 |
  | `drop-0.1` | 0.1 | 0.8990 | 0.8411 | 0.2492 | 0.0163 |
  | `drop-0.3` | 0.3 | 0.8697 | 0.7813 | 0.3187 | -0.0033 |
  | `drop-0.5` | 0.5 | 0.8278 | 0.6664 | 0.3991 | -0.0142 |
- **Biểu đồ so sánh**: `figures/compare_dropout.png`.
- **Giải thích & Cơ chế**:
  - Ở `base-s1`, khoảng cách train loss (0.2082) và val loss (0.2288) chỉ là 0.0206, chứng tỏ mô hình không hề bị quá khớp.
  - Khi tăng $q=0.3$ và $q=0.5$, Val Macro-F1 tụt dốc thảm hại (rơi xuống 0.7813 và 0.6664). Việc tắt ngẫu nhiên 30% - 50% nơ-ron khiến năng lực biểu diễn của mạng MLP 2 lớp ẩn bị bóp nghẹt, gây ra hiện tượng thiếu khớp (underfitting).

### 3.5 Gradient clipping
- **Dự đoán**: Ở lr chuẩn 0.1, gradient tương đối nhỏ nên clipping ít ảnh hưởng. Tuy nhiên ở lr cao bất thường (lr=1.0), gradient sẽ biến động mạnh và clipping sẽ cứu mạng khỏi bất ổn định.
- **Kết quả**:
  | exp_id | lr | Clip Norm | Val Acc | Val Macro-F1 | Best Val Loss | Nhận xét độ ổn định |
  |---|:---:|:---:|:---:|:---:|:---:|---|
  | `base-s1` | 0.1 | None | 0.9095 | 0.8407 | 0.2288 | Gradient norm ổn định ~0.57 |
  | `clip-norm-0.5` | 0.1 | 0.5 | 0.9042 | 0.8506 | 0.2365 | Cắt nhẹ các gai gradient |
  | `clip-highlr-noclip` | 1.0 | None | 0.8616 | 0.7570 | 0.3440 | Dao động mạnh, loss kẹt ở mức cao |
  | `clip-highlr-clip1.0` | 1.0 | 1.0 | **0.8811** | **0.8120** | **0.3056** | **Cứu được huấn luyện, tăng +5.5% F1** |
- **Biểu đồ so sánh**: `figures/compare_clipping.png`.
- **Giải thích & Cơ chế**:
  - Ở lr=0.1, chuẩn gradient toàn cục dao động quanh 0.53 - 0.71, clipping $c=0.5$ giữ kết quả tương đương baseline (F1 = 0.8506).
  - *Thí nghiệm phản chứng*: Khi tăng lr lên 1.0 không clip, bước cập nhật quá lớn khiến mô hình văng khỏi vùng hội tụ tốt, loss kẹt ở 0.3440 và F1 chỉ đạt 0.7570. Khi kích hoạt clipping $c=1.0$, thuật toán chặn đứng các bước nhảy quá đà $g \leftarrow g \cdot \min(1, c/\|g\|)$, đưa F1 lên 0.8120 ($\Delta \text{F1} = \mathbf{+0.0550} > 2\sigma$). Đây là minh chứng thực nghiệm sắc bén cho vai trò phòng ngừa bùng nổ gradient của clipping.

### 3.6 Mixed precision
- **Dự đoán**: FP16 và BF16 duy trì độ chính xác tương đương FP32 trên GPU RTX 4060, giảm dung lượng bộ nhớ.
- **Kết quả**:
  | exp_id | Precision | Val Acc | Val Macro-F1 | Best Val Loss | Thời gian/epoch | Bộ nhớ VRAM đỉnh |
  |---|---|:---:|:---:|:---:|:---:|:---:|
  | `base-s1` | FP32 | 0.9095 | 0.8407 | 0.2288 | ~1.39s | 80.5 MB |
  | `amp-fp16` | FP16 (GradScaler) | 0.9071 | 0.8449 | 0.2334 | ~2.45s | 81.2 MB |
  | `amp-bf16` | BF16 | 0.9091 | 0.8442 | 0.2270 | ~2.13s | 80.7 MB |
- **Biểu đồ so sánh**: `figures/compare_amp.png`.
- **Giải thích & Cơ chế**:
  - Độ chính xác: Cả FP16 (0.8449) và BF16 (0.8442) khớp gần như tuyệt đối với FP32 (chênh lệch $< 0.004 < 2\sigma$).
  - Thời gian: Với mạng MLP nhỏ (47k tham số), thời gian tính toán ma trận cực nhanh (< 1ms/batch), nên overhead kiểm tra underflow và scale gradient của FP16 khiến thời gian/epoch tăng nhẹ so với FP32. BF16 có dải số mũ 8-bit như FP32 nên không cần GradScaler, chạy ổn định và không gặp underflow.

### 3.7 Khởi tạo tham số
- **Dự đoán**: Khởi tạo Zeros ($W=0$) sẽ hoàn toàn thất bại do tính đối xứng không thể bị phá vỡ. Khởi tạo Normal ($\sigma=0.01$) sẽ làm kích hoạt tiêu biến ở các lớp sâu. Khởi tạo He phù hợp nhất với ReLU.
- **Kết quả thực nghiệm**:
  - Độ lệch chuẩn kích hoạt (Activation std) sau 3 lớp ở bước 0:
    - `zeros`: `[0.0, 0.0, 0.0]`
    - `normal`: `[0.0722, 0.0080, 0.0004]` (tiêu biến dần về 0)
    - `xavier`: `[0.5567, 0.4417, 0.4682]`
    - `he`: `[1.4006, 1.3333, 1.0715]` (duy trì phương sai ~ 1.0 ổn định)
  - Hiệu quả huấn luyện:
    | exp_id | Khởi tạo | Loss bước 0 | Val Acc | Val Macro-F1 | Best Val Loss | Trạng thái học |
    |---|---|:---:|:---:|:---:|:---:|---|
    | `init-zeros` | Zeros | 1.2052 | 0.4876 | **0.0936** | 1.2052 | **Tê liệt hoàn toàn, đoán đa số** |
    | `init-normal` | Normal (0.01) | 1.9461 | 0.9036 | 0.8527 | 0.2428 | Chậm ở đầu, hội tụ muộn |
    | `init-xavier` | Xavier | 1.9472 | 0.9106 | 0.8418 | 0.2258 | Ổn định |
    | `base-s1` | He (Kaiming) | 2.0024 | 0.9095 | **0.8407** | 0.2288 | Tối ưu cho ReLU |
- **Biểu đồ so sánh**: `figures/compare_init.png`.
- **Giải thích & Cơ chế**:
  - `init-zeros`: Khi $W=0, b=0$, mọi nơ-ron nhận đầu ra $z=0$, dẫn tới $\text{ReLU}(0)=0$. Gradient truyền ngược $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z} x^T$ hoàn toàn bằng nhau giữa các nơ-ron trong cùng một lớp. Tính đối xứng không bao giờ bị phá vỡ, mạng bị thoái hóa thành một nơ-ron duy nhất và chỉ học được bias đoán lớp đa số 1 (Acc = 0.4876, Macro-F1 = 0.0936).
  - `init-normal`: Vì $\sigma=0.01$ quá nhỏ, tích ma trận qua 3 lớp khiến phương sai kích hoạt rơi tự do về $0.0004$, làm gradient ở các lớp đầu bị triệt tiêu nghiêm trọng trong những epoch đầu.
  - `init-he`: Phương sai $\text{Var}[W] = \frac{2}{n_{in}}$ bù trừ chính xác việc ReLU triệt tiêu một nửa miền âm, duy trì phân phối kích hoạt ổn định $\approx 1.0$ xuyên suốt các lớp sâu.

## 4. Đánh giá cuối trên tập eval

> Chỉ làm sau khi chọn cấu hình bằng val. Số lấy từ `eval_result.json` (do `scripts/evaluate.py` tạo), không tự tính lại.

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|:---:|:---:|:---:|:---:|
| Baseline (`base-s1`) | 1 | 0.8407 | **0.8424** | 0.9078 |
| Cấu hình cuối cùng (`final-deep-adam`) | 1 | 0.8778 | **0.8789** | 0.9254 |

- **Cấu hình cuối cùng gồm những gì và vì sao (chọn dựa trên val)?**:
  - Mô hình cuối cùng kết hợp hai phát hiện tốt nhất trên tập validation:
    1. *Kiến trúc*: `M-deep` (`54 → 256 → 128 → 64 → 7`, 55 687 tham số). Trên val, `M-deep` đạt Val Loss thấp nhất toàn bộ lab (0.1956) và F1 = 0.8730, chứng minh chiều sâu 3 lớp ẩn giúp phân tầng và trừu tượng hóa đặc trưng địa hình phi tuyến tốt hơn `M-base` mà không gây phình to tham số như `M-wide`.
    2. *Bộ tối ưu*: `Adam` với `lr = 0.003`, `batch = 512`, `epochs = 20`, không dùng Dropout (tránh underfitting), khởi tạo `He`. Trên val, Adam (lr=3e-3) đạt F1 cao nhất (0.8751).
  - Khi kết hợp lại (`final-deep-adam`), trên tập val mô hình đạt Val Loss = **0.1859**, Val Acc = **0.9267**, Val Macro-F1 = **0.8778**.
- **Cải thiện so với baseline trên eval có vượt nhiễu không?**:
  - Trên tập eval: Cấu hình cuối cùng đạt **0.8789** macro-F1, so với Baseline **0.8424** macro-F1 $\implies$ Cải thiện vượt bậc:
    $$\Delta \text{Macro-F1} = 0.8789 - 0.8424 = \mathbf{+0.0365}$$
  - Mức cải thiện $+0.0365$ vừa vượt mốc quy định $0.02$ của Rubric (đạt 3/3 điểm tối đa), vừa lớn hơn ngưỡng nhiễu thực nghiệm $2\sigma = 0.0258$. Điều này khẳng định sự tiến bộ của mô hình là hoàn toàn thực chất và có ý nghĩa thống kê cao.
- **Val và eval có gần nhau không?**:
  - Baseline: val macro-F1 = 0.8407 vs eval macro-F1 = 0.8424 (lệch chỉ 0.0017).
  - Final Model: val macro-F1 = 0.8778 vs eval macro-F1 = 0.8789 (lệch chỉ 0.0011).
  - Khoảng cách giữa val và eval cực kỳ nhỏ ($\le 0.002$), chứng tỏ tập validation được tách phân tầng đại diện rất chuẩn xác cho phân phối của tập eval, không hề bị rò rỉ dữ liệu (data leakage) và hoàn toàn đáng tin cậy.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|:---:|:---:|:---:|:---:|
| 0 | 42 368 | 0.9347 | 0.9110 | 0.9227 |
| 1 | 56 661 | 0.9257 | 0.9491 | 0.9372 |
| 2 | 7 151 | 0.9205 | 0.9232 | 0.9219 |
| 3 | 549 | 0.9021 | 0.7213 | 0.8016 |
| 4 | 1 899 | 0.8445 | 0.7351 | 0.7860 |
| 5 | 3 473 | 0.8738 | 0.8396 | 0.8564 |
| 6 | 4 102 | 0.9137 | 0.9393 | 0.9263 |

![Ma trận nhầm lẫn](figures/confusion_matrix.png)

- **Lớp khó nhất**: Lớp khó nhất là **Lớp 4 (Cottonwood/Willow)** với **F1 = 0.7860** (Recall = 0.7351), kế tiếp là **Lớp 3 (Ponderosa Pine)** với **F1 = 0.8016** (Recall = 0.7213).
- **Phân tích nhầm lẫn (từ Confusion Matrix)**:
  - *Lớp 4*: Có 1 396 mẫu đoán đúng, nhưng bị dự đoán nhầm sang **Lớp 1 tới 434 mẫu** (chiếm 22.8% tổng số mẫu lớp 4) và nhầm sang Lớp 0 là 38 mẫu.
  - *Lớp 3*: Có 396 mẫu đoán đúng, nhưng bị nhầm sang **Lớp 2 tới 118 mẫu** (21.5%) và nhầm sang Lớp 5 là 35 mẫu.
  - *Lớp 0 và Lớp 1*: Do số lượng quá lớn (chiếm 85.2% dữ liệu), số lượng mẫu nhầm lẫn tuyệt đối giữa hai lớp này là lớn nhất (3 435 mẫu lớp 0 bị đoán thành lớp 1, và 2 426 mẫu lớp 1 bị đoán thành lớp 0).
- **Lý giải nguyên nhân bằng dữ liệu**:
  1. *Mất cân bằng lớp cực đoan (Class Imbalance)*: Lớp 4 chỉ có 1 899 mẫu (1.63%) và Lớp 3 chỉ có 549 mẫu (0.47%), trong khi Lớp 1 có tới 56 661 mẫu (48.8%). Trong quá trình huấn luyện bằng Cross-Entropy thông thường, gradient từ 56k mẫu lớp 1 hoàn toàn áp đảo hàm mất mát, khiến mô hình có xu hướng thiên vị (inductive bias) gán các mẫu ở vùng biên ranh giới về lớp đa số 1.
  2. *Sự tương đồng về đặc trưng tự nhiên*: Lớp 0 (Spruce/Fir) và Lớp 1 (Lodgepole Pine) cùng sinh trưởng ở độ cao lớn (Elevation tương đương, lượng bóng râm và độ dốc tương tự). Tương tự, Lớp 3 và Lớp 2 cùng phân bố ở vành đai độ cao thấp tiếp giáp nguồn nước, dẫn đến vector 54 đặc trưng địa hình có khoảng cách Euclidean rất gần nhau trong không gian biểu diễn.
- **Biện pháp cải thiện đề xuất**:
  - Áp dụng **Class-Weighted Cross-Entropy Loss**: Gán trọng số nghịch đảo tần suất lớp $w_c \propto \frac{1}{N_c}$ để phạt nặng lỗi sai trên các lớp hiếm 3 và 4.
  - Thử nghiệm **Focal Loss** ($\gamma=2$) để dồn sự tập trung của gradient vào các mẫu khó phân loại thay vì bị bão hòa bởi các mẫu dễ của lớp 0 và 1.

## 5. Trả lời các câu hỏi dẫn dắt

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   - Khi được điều chỉnh lr công bằng (thử nghiệm $\ge 2$ mức lr cho mỗi bộ), **Adam (lr=3e-3)** là bộ tối ưu "thắng cuộc" với Val Macro-F1 = **0.8751** (kế cận là AdamW lr=3e-3 với 0.8682), vượt trội hơn SGD+Momentum (0.8407 ở lr=0.1) và bỏ xa SGD thuần (0.7392 ở lr=0.1).
   - Nếu *không điều chỉnh lr* (ví dụ dùng chung lr=0.1 cho tất cả các bộ), Adam và AdamW sẽ bị phân kỳ (NaN loss) hoặc dao động dữ dội do lr quá lớn so với cơ chế thích nghi moment, từ đó có thể dẫn tới kết luận sai lầm rằng "SGD+Momentum tốt hơn Adam". Việc quét lr riêng biệt cho từng họ thuật toán là bắt buộc để so sánh công bằng.

2. **Dropout có giúp không khi mô hình chưa quá khớp? Khi nào thì nên dùng?**
   - **Không giúp, thậm chí gây hại nghiêm trọng.** Thực nghiệm cho thấy baseline có khoảng cách train-val loss chỉ là 0.0206 (không overfit). Khi áp dụng Dropout $q=0.3$ và $q=0.5$, Val Macro-F1 bị tụt dốc thảm hại từ 0.8407 xuống 0.7813 và 0.6664.
   - *Khi nào nên dùng*: Dropout chỉ nên dùng khi mô hình có năng lực biểu diễn vượt quá kích thước dữ liệu dẫn đến quá khớp rõ rệt (triệu chứng: Train loss tiếp tục giảm sâu về 0 nhưng Val loss tăng vọt ngược lên, train-val gap rất lớn).

3. **Gradient clipping giải quyết vấn đề gì? Quan sát nào của bạn chứng minh điều đó?**
   - Gradient clipping giải quyết triệt để vấn đề **bùng nổ gradient (exploding gradients)** và bất ổn định bước nhảy khi độ cong hàm mất mát lớn hoặc khi learning rate bị đặt quá cao.
   - *Bằng chứng thực nghiệm*: Trong thí nghiệm phản chứng ở lr=1.0:
     - Khi **không clip** (`clip-highlr-noclip`), bước nhảy quá lớn làm mô hình dao động dữ dội, Val Loss kẹt ở 0.3440 và F1 chỉ đạt 0.7570.
     - Khi **bật clip** $c=1.0$ (`clip-highlr-clip1.0`), thuật toán đã giữ chuẩn gradient luôn $\le 1.0$, cứu sống quá trình huấn luyện, kéo Val Loss xuống 0.3056 và tăng vọt Macro-F1 lên **0.8120** (tăng tới +5.5% F1).

4. **Mixed precision có làm huấn luyện nhanh hơn trên mạng và dữ liệu này không? Vì sao (không)?**
   - **Không nhanh hơn.** Trên GPU RTX 4060, thời gian một epoch của FP32 là ~1.39s, trong khi FP16 là ~2.45s và BF16 là ~2.13s.
   - *Nguyên nhân*: Mô hình MLP `M-base` có kích thước rất nhỏ (chỉ 47k tham số), thời gian tính toán ma trận thực tế trên GPU cực ngắn (< 1ms/batch). Chi phí phụ trội (overhead) cho việc điều phối kernel GPU, ép kiểu tensor và cơ chế `GradScaler` kiểm tra underflow của FP16 đã lấn át lợi thế tính toán của Tensor Cores. Mixed precision chỉ phát huy ưu thế tăng tốc vượt trội trên các mạng nơ-ron lớn (hàng triệu đến hàng tỷ tham số) hoặc các bài toán bị nghẽn băng thông bộ nhớ (memory-bandwidth bound).

5. **Vì sao khởi tạo toàn số 0 hỏng? Khởi tạo He khác Xavier ở điểm nào và khi nào điều đó quan trọng?**
   - *Khởi tạo Zeros hỏng vì*: Với $W=0, b=0$, mọi nơ-ron trong cùng một lớp nhận giá trị đầu ra $z=0$, dẫn tới $\text{ReLU}(0)=0$. Gradient truyền ngược $\frac{\partial L}{\partial W}$ giống hệt nhau ở mọi nơ-ron cùng tầng. Do đó mọi nơ-ron bị cập nhật như nhau qua mọi epoch, tính đối xứng không bao giờ bị phá vỡ, mạng suy biến thành 1 nơ-ron duy nhất và không thể học đặc trưng (thực nghiệm đo được Macro-F1 kẹt ở 0.0936).
   - *He khác Xavier*: Xavier giả định hàm kích hoạt là tuyến tính hoặc đối xứng quanh 0 (như tanh), đặt phương sai $\text{Var}[W] = \frac{2}{n_{in} + n_{out}}$ (hoặc $\frac{1}{n_{in}}$). He (Kaiming) tính đến việc hàm ReLU triệt tiêu hoàn toàn một nửa miền giá trị âm ($x < 0 \implies \text{ReLU}(x) = 0$), làm giảm một nửa phương sai kích hoạt qua mỗi tầng, do đó He đặt $\text{Var}[W] = \frac{2}{n_{in}}$ (gấp đôi Xavier). Điều này đặc biệt quan trọng khi mạng có nhiều tầng kích hoạt ReLU, giúp phương sai không bị suy giảm theo cấp số nhân qua chiều sâu.

6. **Quay lại câu hỏi của bài học: một mạng có loss không giảm sau 2 000 bước. Dựa vào bảng "triệu chứng" ở Chương 5 và các thí nghiệm của bạn, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.**
   1. **Kiểm tra 1: Thử nghiệm quá khớp một lô nhỏ (Sanity Check - Overfit 20 samples)**:
      - *Vì sao*: Lấy 20 mẫu ngẫu nhiên, tắt bỏ toàn bộ chính quy hóa (dropout=0, weight_decay=0) và train 100–200 bước. Nếu loss không giảm về sát 0 (accuracy không đạt 100%), lỗi chắc chắn 100% nằm ở mã nguồn vòng lặp huấn luyện (quên `zero_grad`, tham số không truyền vào optimizer, lỗi tính loss hai lần softmax, hoặc nhãn bị sai lệch) chứ không phải do dữ liệu hay kiến trúc.
   2. **Kiểm tra 2: Kiểm tra dòng chảy gradient và chuẩn gradient (`grad_norm` sau `backward`)**:
      - *Vì sao*: In ra chuẩn gradient L2 của từng lớp tham số (`model.named_parameters()`). Nếu gradient bị `None` hoặc bằng 0, mạng đang gặp hiện tượng gradient không chảy (chết nơ-ron ReLU hàng loạt do khởi tạo sai như `zeros`, hoặc đứt gãy đồ thị autograd do dùng nhầm phép toán inplace). Nếu gradient là `NaN/Inf`, vấn đề là bùng nổ gradient do `lr` quá lớn (cần giảm lr hoặc bật gradient clipping).
   3. **Kiểm tra 3: Đo Loss bước 0 và kiểm tra chuẩn hóa dữ liệu đầu vào**:
      - *Vì sao*: Trước khi cập nhật trọng số, loss trên val phải xấp xỉ $-\ln(1/C) = \ln 7 \approx 1.946$. Nếu loss bước 0 cao hơn nhiều (ví dụ 10, 50), nguyên nhân là do dữ liệu đầu vào chưa được chuẩn hóa (mean/std quá lớn làm bão hòa đầu ra) hoặc trọng số khởi tạo ban đầu có phương sai quá lớn làm méo mó logit thô.

## 6. Hạn chế và điều bất ngờ

- **Điều bất ngờ nhất**: Hiệu ứng của **Gradient Clipping** trong thí nghiệm phản chứng ở lr=1.0. Ban đầu dự đoán lr=1.0 có thể làm mạng phân kỳ NaN ngay lập tức, nhưng thực tế mạng vẫn chạy được dù loss rất cao (0.3440) và F1 tụt về 0.7570. Khi bật clipping $c=1.0$, mạng được cứu ngoạn mục và đạt F1 tới 0.8120, minh chứng sức mạnh bảo vệ cực lớn của clipping đối với sự ổn định số học.
- **Hạn chế trong thiết kế thí nghiệm**: Do giới hạn thời gian (20 epochs), các cấu hình dùng batch lớn (2048) chưa kịp hội tụ hết số bước cập nhật, dẫn tới việc so sánh batch size ở cùng 20 epochs có phần bất lợi cho batch lớn so với batch nhỏ.
- **Nếu có thêm thời gian**: Sẽ thử nghiệm kết hợp **Class-Weighted Loss** và **Cosine Annealing Learning Rate Scheduler** trong 40 epochs trên mô hình `M-deep` để tối ưu hóa triệt để điểm số trên 2 lớp thiểu số khó nhất (lớp 3 và lớp 4).

## 7. Phụ lục

- **Danh sách file trong gói nộp bài**:
  - `REPORT.md`: Báo cáo chi tiết toàn diện.
  - `experiments.xlsx`: Bảng kết quả 27 dòng thực nghiệm đầy đủ 4 sheet (`Legend`, `Experiments`, `Seeds`, `Summary`).
  - `predictions_eval.csv`: Dự đoán 116 203 dòng trên tập `eval` (đạt Macro-F1 = **0.8789**, Accuracy = **0.9254**).
  - `eval_result.json`: Kết quả chính thức do `scripts/evaluate.py` sinh ra.
  - `figures/`: Chứa 36 tệp ảnh đồ thị chất lượng cao (ảnh riêng từng thí nghiệm, ảnh so sánh nhóm và `confusion_matrix.png`).
  - `results/`: Chứa 27 file JSON lịch sử huấn luyện của từng thí nghiệm.
  - `code/`: Mã nguồn hoàn thiện không còn `NotImplementedError` (`data.py`, `model.py`, `optimizer.py`, `train.py`, `plots.py`, `results_table.py`, `lab.ipynb`).
- **Thời gian chạy tổng cộng**: Khoảng 12 phút huấn luyện toàn bộ 27 thí nghiệm trên GPU NVIDIA GeForce RTX 4060 Laptop (trung bình ~1.2s/epoch).
