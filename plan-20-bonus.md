# Plan lấy full cap +20 bonus — Day 22 Track 3 bằng Google Colab

Plan này tập trung vào **bonus rigor add-ons trong `rubric.md`** và giả định bạn tiếp tục chạy bằng **Google Colab** với notebook chính:

- `colab/Lab22_DPO_T4.ipynb`

Mục tiêu là lấy đủ **core 100** trước, rồi gom bonus theo cách an toàn để chạm **mức cap +20**.

---

## 1. Mục tiêu bonus: chọn tổ hợp vượt cap để có buffer

Theo `rubric.md`, các bonus là:

- **β-sweep mini-experiment**: +6
- **HuggingFace Hub push**: +5
- **GGUF release published**: +3
- **MMLU full coverage**: +3
- **Weights & Biases run link**: +2
- **Cross-judge comparison**: +4

Tổng tối đa liệt kê là **+23**, nhưng bị **cap ở +20**.

### Tổ hợp đề xuất

Ưu tiên làm đủ các mục sau:

1. **β-sweep mini-experiment** (+6)
2. **HuggingFace Hub push** (+5)
3. **GGUF release published** (+3)
4. **MMLU full coverage** (+3)
5. **Weights & Biases public run link** (+2)
6. **Cross-judge comparison** (+4) — làm như buffer an toàn

=> Nếu 1 mục bị thiếu hoặc làm chưa đẹp, bạn vẫn còn khả năng đạt **cap +20**.

---

## 2. Nguyên tắc chạy: core trước, bonus sau

Không nên nhảy vào bonus ngay từ đầu.

### Thứ tự bắt buộc

1. Chạy xong **core pipeline** trong `colab/Lab22_DPO_T4.ipynb`
2. Tải artifacts về hoặc lưu an toàn trên Colab/Drive
3. Xác nhận các artifact core đã có đủ
4. Mới bắt đầu chạy các bonus add-ons

### Artifact core cần có trước khi làm bonus

- `adapters/sft-mini/adapter_config.json`
- `adapters/dpo/adapter_config.json`
- `adapters/dpo/dpo_metrics.json`
- `data/pref/train.parquet`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `data/eval/benchmark_results.json`
- `gguf/*Q4_K_M*.gguf`
- `submission/screenshots/02-sft-loss.png`
- `submission/screenshots/03-dpo-reward-curves.png`
- `submission/screenshots/04-side-by-side-table.png`
- `submission/screenshots/07-benchmark-comparison.png`

---

## 3. Workflow Colab đề xuất

## Phase A — Chạy core lab

### Bước 1 — Mở đúng notebook

Chạy file:

- `colab/Lab22_DPO_T4.ipynb`

### Bước 2 — Chọn runtime đúng

Trong Colab:

- `Runtime -> Change runtime type -> T4 GPU`

### Bước 3 — Chạy toàn bộ pipeline core

Run các stage theo thứ tự NB1 → NB6.

### Bước 4 — Package artifact cuối notebook

Chạy cell package/download ở cuối notebook để tải zip artifact về máy.

---

## Phase B — Làm bonus add-ons

## 4. Bonus 1: β-sweep mini-experiment (+6)

### Mục tiêu

Chạy DPO với các giá trị:

- `β = 0.05`
- `β = 0.1`
- `β = 0.5`

### Cần thu thập

- reward gap cho từng β
- nếu được thì thêm win-rate/summary cho từng β
- 1 biểu đồ tổng hợp kiểu `bonus-beta-sweep.png`
- phần diễn giải **≥ 100 từ**

### Kết quả tối thiểu nên có

- bảng so sánh 3 giá trị β
- ảnh biểu đồ bonus
- nhận xét: β nào ổn nhất, β nào quá mạnh/quá yếu

---

## 5. Bonus 2: Hugging Face Hub push (+5)

### Username Hugging Face

- `kinchunnono`

### Yêu cầu bảo mật

Không hardcode token thật vào repo.

Trong Colab chỉ chuẩn bị cell với biến placeholder:

```python
HUGGINGFACE_TOKEN = ""
```

Bạn sẽ tự điền token write permission vào cell đó khi chạy Colab.

### Mục tiêu push

Push DPO adapter lên repo kiểu:

- `kinchunnono/lab22-dpo-vn`

### Cần có trên HF model card

- base model
- dataset SFT
- dataset preference
- hyperparameters chính
- benchmark/eval summary
- link về repo GitHub của lab

---

## 6. Bonus 3: GGUF release published (+3)

### Mục tiêu

Publish GGUF lên Hugging Face, tối thiểu có:

- `Q4_K_M`
- `Q5_K_M`

### Bằng chứng cần có

- trang HF repo/release hiển thị cả 2 file quantization
- screenshot hoặc link công khai

---

## 7. Bonus 4: MMLU full coverage (+3)

### Mục tiêu

Rerun benchmark với:

- `LIMIT_MMLU = 14000`

### Cần báo cáo

- kết quả MMLU sampled trước đó
- kết quả MMLU full mới
- chênh lệch giữa sampled vs full
- nhận xét xem sampled có đại diện tốt không

---

## 8. Bonus 5: Weights & Biases public run link (+2)

### Mục tiêu

Bật tracking public cho ít nhất các run DPO/β-sweep.

### Cần có

- link public W&B
- ảnh chụp màn hình thấy curves rõ ràng
- run name dễ đọc, ví dụ theo từng β

---

## 9. Bonus 6: Cross-judge comparison (+4)

### Mục tiêu

Chạy judge bằng **cả hai** model:

- `gpt-4o-mini`
- `claude-haiku`

Áp dụng cho:

- NB4 side-by-side judge
- nếu làm được, thêm NB6 AlpacaEval-lite judge

### Cần tổng hợp

- prompt nào 2 judge đồng ý
- prompt nào bất đồng
- disagreement rate
- nhận xét judge bias

---

## 10. Sau khi chạy xong bonus

Sau khi hoàn tất bonus trên Colab:

1. Tải zip artifact về máy
2. Gửi zip đó cho mình
3. Mình sẽ:
   - giải nén
   - đọc benchmark JSON / outputs / screenshots / notebook outputs
   - kiểm tra đã đủ benchmark chưa
   - kiểm tra bonus evidence đã đủ mạnh chưa
   - viết hoàn chỉnh `submission/REFLECTION.md` cho bạn
   - chỉ ra chỗ nào còn thiếu trước khi submit

---

## 11. Checklist ảnh cần chụp step-by-step

## A. Ảnh core bắt buộc

1. **GPU setup**
   - chụp cell hiện `nvidia-smi` hoặc tên `Tesla T4`
2. **SFT loss curve**
   - chụp biểu đồ loss ở NB1
3. **DPO reward curves**
   - chụp đầy đủ `chosen_rewards`, `rejected_rewards`, gap
4. **Side-by-side table**
   - chụp bảng 8 prompts × 2 model outputs
5. **Judge output / manual rubric**
   - chụp verdict rõ ràng
6. **GGUF smoke test**
   - chụp cell load GGUF + output tiếng Việt
7. **Benchmark comparison**
   - chụp 4-bar chart có delta

## B. Ảnh bonus nên chụp thêm

8. **β-sweep chart**
9. **HF adapter page**
10. **HF GGUF page**
11. **W&B public run page**
12. **MMLU full result cell**
13. **Cross-judge summary**

---

## 12. Định nghĩa done cho plan này

Plan này được coi là hoàn thành khi:

- core 100 đã chạy xong trên Colab
- có zip artifact đầy đủ
- có evidence cho ít nhất tổ hợp bonus đủ chạm cap +20
- mình đã kiểm benchmark sau khi bạn gửi zip
- mình đã viết xong report lab hoàn chỉnh cho bạn
- mình đã hướng dẫn lại cách chụp ảnh theo đúng từng mục submit

---

## 13. Chiến lược an toàn nhất để chạm cap +20

Nếu muốn tối ưu công sức / tỷ lệ thành công, thứ tự ưu tiên làm bonus là:

1. β-sweep
2. HF adapter push
3. GGUF publish
4. MMLU full
5. W&B link
6. Cross-judge

Đây là đường đi an toàn vì:

- β-sweep cho nhiều điểm nhất
- HF + GGUF tận dụng luôn artifact bạn đã train
- MMLU full là benchmark extension tự nhiên
- W&B và cross-judge tăng độ chắc chắn nếu một bonus khác chưa đẹp
