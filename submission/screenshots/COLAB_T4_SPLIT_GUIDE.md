# Colab T4 split guide

Dùng guide này nếu bạn chạy theo 2 notebook:
- `colab/Lab22_DPO_T4_Part1.ipynb`
- `colab/Lab22_DPO_T4_Part2.ipynb`

## Thứ tự chạy

1. Mở `Lab22_DPO_T4_Part1.ipynb` và chạy từ đầu đến cuối.
2. Ở cell cuối Part 1, tải `day22-colab-part1.zip` về máy.
3. Mở runtime mới.
4. Mở `Lab22_DPO_T4_Part2.ipynb`.
5. Chạy cell restore gần đầu notebook, upload `day22-colab-part1.zip`.
6. Chạy tiếp đến cuối notebook và tải `day22-colab-part2.zip`.

## Ảnh bắt buộc: chụp ở NB nào, bước nào

| Ảnh | Notebook | NB | Bước cần chụp | Chụp cái gì |
|---|---|---|---|---|
| `01-setup-gpu.png` | Part 1 | Setup | `## A. Colab setup` → cell `# Probe GPU` | Thấy tên GPU và VRAM, ví dụ `T4` / dung lượng RAM GPU |
| `02-sft-loss.png` | Part 1 | NB1 | `### 3a. Plot loss curve` | Biểu đồ loss SFT giảm dần qua 1 epoch |
| `03-dpo-reward-curves.png` | Part 1 | NB3 | `## 5. Plot reward curves — THE diagnostic` | Thấy riêng `chosen_rewards`, `rejected_rewards`, và gap |
| `04-side-by-side-table.png` | Part 2 | NB4 | `## 4. Side-by-side table` hoặc `### 4a. Render as a markdown table image` | Bảng ≥ 8 prompt, có cả output SFT và SFT+DPO |
| `05-judge-output.png` | Part 2 | NB4 | `## 5. Optional: API judge` | Chụp verbatim verdict của judge cho ít nhất 3 prompt |
| `05-manual-rubric.png` | Part 2 | NB4 | `### 5a. Manual rubric fallback` | Nếu không dùng API judge, chụp cell manual rubric đã điền; nếu crop được, nên kèm luôn phần `## 6. Win/loss/tie summary` ở cùng ảnh hoặc chụp thêm ảnh phụ |
| `06-gguf-smoke.png` | Part 2 | NB5 | `### 4a. Smoke prompt + response` | Phải thấy tên file `lab22-dpo-Q4_K_M.gguf`, prompt tiếng Việt, response tiếng Việt coherent |
| `07-benchmark-comparison.png` | Part 2 | NB6 | `## 6. Aggregate + 4-bar comparison plot` | Biểu đồ 4 benchmark với SFT vs DPO và delta annotate |

## Chi tiết nên làm theo

### Part 1

#### 1) `01-setup-gpu.png`
- Chụp ngay sau cell `# Probe GPU`.
- Ảnh cần thấy rõ:
  - tên GPU
  - VRAM
  - đây là runtime bạn thực sự dùng

#### 2) `02-sft-loss.png`
- Chụp sau NB1, bước `### 3a. Plot loss curve`.
- Crop vừa khung biểu đồ, thấy đủ trục và đường loss.

#### 3) `03-dpo-reward-curves.png`
- Chụp sau NB3, bước `## 5. Plot reward curves — THE diagnostic`.
- Ảnh cần thấy:
  - `chosen_rewards`
  - `rejected_rewards`
  - reward gap
  - legend và trục

#### 4) Tải zip Part 1
- Chạy cell cuối: `## Final — Package Part 1 artifacts and auto-download`.
- Đợi tải xong `day22-colab-part1.zip` rồi mới chuyển sang Part 2.

### Part 2

#### 5) Restore artifact trước khi chạy NB4
- Chạy cell `## Restore Part 1 handoff zip`.
- Upload `day22-colab-part1.zip`.
- Chỉ khi cell báo restore xong mới chạy tiếp.

#### 6) `04-side-by-side-table.png`
- Chụp ở NB4, bước `## 4. Side-by-side table` hoặc ảnh render ở `### 4a`.
- Nếu bảng dài hơn 1 màn hình, có thể chụp 2 ảnh `04a` và `04b`.

#### 7) `05-judge-output.png` hoặc `05-manual-rubric.png`
- Nếu có API key, chụp ở `## 5. Optional: API judge` và nhớ thấy verbatim verdict của judge cho ít nhất 3 prompt.
- Nếu không có API key, điền cell `### 5a. Manual rubric fallback`, rerun cell đó để lưu `judge_results.json`, rồi chụp cell manual rubric đã điền; nếu crop được, nên để `## 6. Win/loss/tie summary` xuất hiện cùng ảnh hoặc chụp thêm ảnh phụ hỗ trợ.
- Nhớ crop để không lộ API key.

#### 8) `06-gguf-smoke.png`
- Chụp ở NB5, bước `### 4a. Smoke prompt + response`.
- Ảnh nên thấy đủ 4 thứ:
  1. dòng `Loading: lab22-dpo-Q4_K_M.gguf`
  2. prompt tiếng Việt
  3. response tiếng Việt coherent
  4. nếu có, usage / token stats

#### 9) `07-benchmark-comparison.png`
- Chụp ở NB6, bước `## 6. Aggregate + 4-bar comparison plot`.
- Ảnh cần thấy đủ 4 benchmark: IFEval, GSM8K, MMLU, AlpacaEval-lite.

#### 10) Tải zip Part 2
- Chạy cell cuối: `## Final — Package Part 2 artifacts and auto-download`.
- Notebook giờ sẽ chỉ pass cell này khi đã có file chuẩn rubric: `gguf/lab22-dpo-Q4_K_M.gguf`.

## Mẹo nộp bài

- Crop chặt vào output cần chấm, đừng chụp full desktop.
- Với NB4 và NB6, ưu tiên ảnh thấy được tiêu đề plot / table, legend, và phần output chính.
- Nếu tên file ảnh khác các tên gợi ý ở đây thì vẫn được, miễn bạn map rõ trong `submission/REFLECTION.md`.
