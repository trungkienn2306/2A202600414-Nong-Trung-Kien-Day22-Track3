# Plan đạt 100 điểm — Day 22 Track 3 DPO/ORPO Alignment Lab bằng Google Colab

Tài liệu này là plan làm bài theo `rubric.md`, nhưng đã chỉnh lại theo thực tế máy local hiện tại: GPU local là **Intel Iris Xe Graphics shared memory**, không phải NVIDIA CUDA GPU, nên **không nên chạy training local**. Chiến lược chính để lấy 100 điểm core là chạy pipeline trên **Google Colab T4**, sau đó tải artifacts về repo này để hoàn thiện reflection và submit.

## 1. Kết luận phần cứng

### Máy local hiện tại

Ảnh GPU cho thấy:

- GPU: Intel Iris Xe Graphics.
- GPU memory: shared memory khoảng 7.9 GB.
- Không phải NVIDIA GPU.
- Không có CUDA.
- Không có VRAM rời.

Theo `HARDWARE-GUIDE.md`, DPO cần GPU NVIDIA vì DPO load policy/reference model và cần bitsandbytes 4-bit. CPU hoặc Intel iGPU không phù hợp cho full lab.

### Quyết định

- **Không chạy full pipeline local.**
- **Chạy trên Google Colab T4.**
- Local chỉ dùng để:
  - quản lý repo/git;
  - đọc/sửa markdown;
  - viết `REFLECTION.md`;
  - lưu artifacts tải từ Colab về;
  - chạy verify nhẹ nếu môi trường local đủ dependency, hoặc verify ngay trên Colab.

## 2. Mục tiêu

Mục tiêu là lấy đủ **100 điểm core** theo `rubric.md` bằng đường Colab T4:

1. Mở hoặc upload repo lên Google Colab.
2. Chọn runtime T4 GPU.
3. Chạy `colab/Lab22_DPO_T4.ipynb` hoặc chạy make targets trong Colab.
4. Sinh đầy đủ artifacts:
   - SFT adapter;
   - preference parquet;
   - DPO adapter;
   - side-by-side eval;
   - GGUF Q4_K_M;
   - benchmark JSON/chart;
   - screenshots.
5. Tải artifacts từ Colab về repo local.
6. Điền `submission/REFLECTION.md` bằng số liệu thật.
7. Chạy `make verify` trên Colab hoặc local.
8. Push repo public và submit URL vào LMS.

## 3. Workflow tổng quan trên Colab

Có 2 cách chạy. Ưu tiên Cách A nếu bạn muốn ít thao tác nhất.

### Cách A — Dùng single-file Colab notebook

Repo có sẵn:

```text
colab/Lab22_DPO_T4.ipynb
```

Các bước:

1. Mở Google Colab.
2. Upload hoặc mở file `colab/Lab22_DPO_T4.ipynb`.
3. Vào menu:

```text
Runtime → Change runtime type → Hardware accelerator → T4 GPU
```

4. Chạy cell đầu kiểm tra GPU.
5. Run all từ đầu đến cuối.
6. Nếu bị timeout hoặc lỗi giữa chừng, chạy lại từng section theo thứ tự NB1 → NB6.

### Cách B — Clone repo trong Colab rồi chạy Makefile

Trong một Colab notebook mới, chạy:

```bash
!git clone https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git
%cd Day22-Track3-DPO-Alignment-Lab
!bash setup-colab.sh
```

Sau đó chạy:

```bash
!make smoke
!make pipeline
!make verify
```

Nếu muốn dễ debug hơn, chạy từng bước:

```bash
!make sft
!make data
!make dpo
!make eval
!make deploy
!make bench
!make verify
```

## 4. Lưu ý quan trọng khi dùng Colab

### 4.1 Chọn đúng runtime

Trước khi chạy setup hoặc notebook, bắt buộc chọn T4:

```text
Runtime → Change runtime type → T4 GPU
```

Kiểm tra bằng cell:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Kết quả mong muốn:

```text
True
Tesla T4
```

Nếu không thấy `Tesla T4`, dừng lại và đổi runtime. Không chạy bằng CPU.

### 4.2 Colab có thể disconnect

DPO + benchmark có thể chạy lâu. Để giảm rủi ro:

- Chạy từng step thay vì Run all nếu mạng yếu.
- Sau mỗi step quan trọng, tải artifact về hoặc copy sang Google Drive.
- Nếu gần hết thời gian runtime, ưu tiên lưu adapters/data/screenshots trước.

### 4.3 Lưu artifact ra Google Drive nếu cần

Có thể mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Sau đó copy artifacts sang Drive:

```bash
!mkdir -p /content/drive/MyDrive/day22-artifacts
!cp -r adapters data gguf submission notebooks /content/drive/MyDrive/day22-artifacts/
```

Cách này giúp tránh mất kết quả nếu Colab disconnect.

## 5. Checklist 100 điểm theo rubric

### 5.1 NB1 — SFT-mini: 15 điểm

Cần đủ:

- [ ] `adapters/sft-mini/adapter_config.json` tồn tại.
- [ ] `adapter_config.json` có `lora_alpha: 32`.
- [ ] `adapter_config.json` có `r: 16`.
- [ ] SFT loss curve giảm qua 1 epoch.
- [ ] Notebook 01 có ít nhất 1 sample generation từ SFT model.
- [ ] Có screenshot loss curve.

Artifacts cần giữ lại:

```text
adapters/sft-mini/adapter_config.json
notebooks/01_sft_mini.ipynb
submission/screenshots/02-sft-loss.png
```

Trên Colab cần chụp:

- output cell hiển thị GPU hoặc training config;
- loss curve;
- sample generation.

### 5.2 NB2 — Preference data: 10 điểm

Cần đủ:

- [ ] `data/pref/train.parquet` tồn tại.
- [ ] Parquet có cột `prompt`.
- [ ] Parquet có cột `chosen`.
- [ ] Parquet có cột `rejected`.
- [ ] Notebook 02 in 3 inspected examples.
- [ ] Mỗi inspected example có token counts.
- [ ] Mỗi inspected example có `chosen != rejected`.

Artifacts cần giữ lại:

```text
data/pref/train.parquet
data/pref/eval.parquet
notebooks/02_preference_data.ipynb
```

### 5.3 NB3 — DPO train: 25 điểm

Đây là phần nhiều điểm nhất.

Cần đủ:

- [ ] `adapters/dpo/adapter_config.json` tồn tại.
- [ ] DPO adapter khác SFT-mini adapter.
- [ ] Có reward gap plot.
- [ ] Reward gap `chosen - rejected` tăng.
- [ ] Plot thể hiện riêng `chosen_rewards`.
- [ ] Plot thể hiện riêng `rejected_rewards`.
- [ ] `submission/REFLECTION.md` §3 diễn giải riêng cả chosen và rejected trajectories.
- [ ] §3 liên hệ deck §3.4, đặc biệt likelihood displacement.

Artifacts cần giữ lại:

```text
adapters/dpo/adapter_config.json
adapters/dpo/dpo_metrics.json
notebooks/03_dpo_train.ipynb
submission/screenshots/03-dpo-reward-curves.png
```

Điểm dễ mất:

- Không được chỉ viết “reward gap tăng nên DPO tốt”.
- Phải mô tả chosen reward đi lên/xuống/dao động.
- Phải mô tả rejected reward đi lên/xuống/dao động.
- Nếu gap tăng do rejected giảm nhanh hơn chosen, cần nói rõ nguy cơ likelihood displacement.

### 5.4 NB4 — Compare and eval: 10 điểm

Cần đủ:

- [ ] Có bảng side-by-side ít nhất 8 prompts × 2 model outputs.
- [ ] Hai model là SFT-only và SFT+DPO.
- [ ] Có 4 helpfulness prompts.
- [ ] Có 4 safety prompts.
- [ ] Có judge call hoặc manual call.
- [ ] Có win/loss/tie summary.

Artifacts cần giữ lại:

```text
data/eval/prompts.json
data/eval/side_by_side.jsonl
data/eval/judge_results.json
notebooks/04_compare_and_eval.ipynb
submission/screenshots/04-side-by-side-table.png
submission/screenshots/05-judge-output.png
```

Nếu không có OpenAI/Anthropic API key, dùng manual rubric mode. Không mất điểm nếu manual rubric có evidence rõ.

### 5.5 NB5 — Merge/deploy GGUF: 10 điểm

Cần đủ:

- [ ] Có file GGUF trong `gguf/`.
- [ ] File đúng target Q4_K_M.
- [ ] File GGUF nhỏ hơn 5 GB.
- [ ] Có llama.cpp hoặc llama-cpp-python smoke test.
- [ ] Smoke test hiển thị filename Q4_K_M.
- [ ] Smoke test sinh output tiếng Việt coherent.

Artifacts cần giữ lại:

```text
gguf/lab22-dpo-Q4_K_M.gguf
notebooks/05_merge_deploy_gguf.ipynb
submission/screenshots/06-gguf-smoke.png
```

Nếu file GGUF tên khác nhưng có Q4_K_M do notebook sinh ra, vẫn giữ lại và trong reflection ghi rõ file nào là GGUF evidence.

### 5.6 NB6 — Benchmark: 10 điểm

Cần đủ:

- [ ] `data/eval/benchmark_results.json` tồn tại.
- [ ] JSON có score cho SFT-only.
- [ ] JSON có score cho SFT+DPO.
- [ ] Có 4 benchmark: IFEval, GSM8K, MMLU, AlpacaEval-lite.
- [ ] Có `07-benchmark-comparison.png`.
- [ ] Chart có absolute scores.
- [ ] Chart có delta annotations.

Artifacts cần giữ lại:

```text
data/eval/benchmark_results.json
notebooks/06_benchmark.ipynb
submission/screenshots/07-benchmark-comparison.png
```

### 5.7 Reproducibility, reflection, verify: 20 điểm

Cần đủ:

- [ ] Repo có thể chạy bằng Colab Run-all hoặc `setup-colab.sh` + `make pipeline`.
- [ ] `submission/REFLECTION.md` có đủ 7 sections.
- [ ] §3 Reward curves analysis viết ít nhất 150 từ.
- [ ] §6 Personal reflection viết ít nhất 150 từ.
- [ ] §7 Benchmark interpretation viết ít nhất 150 từ.
- [ ] §3 diễn giải cả `chosen_rewards` và `rejected_rewards`.
- [ ] §3 nhắc deck §3.4 hoặc likelihood displacement.
- [ ] §7 nói benchmark nào tăng và benchmark nào giảm.
- [ ] §7 nhắc alignment tax theo deck §8.1.
- [ ] `make verify` exit 0.

Lưu ý: template `REFLECTION.md` ghi §3 ≥100 từ, nhưng rubric yêu cầu ≥150 từ cho §3, §6, §7. Làm theo rubric.

## 6. Kế hoạch chạy thực tế trên Colab T4

### Phase 0 — Chuẩn bị trước khi mở Colab

- [ ] Push repo hiện tại lên GitHub hoặc chuẩn bị file zip để upload vào Colab.
- [ ] Đảm bảo repo có các file mới:
  - `PLAN_100_POINTS.md`
  - `instruction.md`
- [ ] Đọc nhanh `instruction.md` để hiểu pipeline.
- [ ] Đọc `rubric.md` và `submission/screenshots/README.md`.

### Phase 1 — Mở Colab và kiểm tra GPU

- [ ] Mở `colab/Lab22_DPO_T4.ipynb`.
- [ ] Chọn T4 GPU.
- [ ] Chạy cell kiểm tra GPU.
- [ ] Chụp screenshot GPU làm `01-setup-gpu.png`.

Nếu dùng clone repo trong Colab:

```bash
!git clone https://github.com/<your-username>/Day22-Track3-DPO-Alignment-Lab.git
%cd Day22-Track3-DPO-Alignment-Lab
!bash setup-colab.sh
!make smoke
```

### Phase 2 — Chạy NB1 SFT-mini

- [ ] Chạy section NB1.
- [ ] Kiểm tra adapter SFT được lưu.
- [ ] Chụp loss curve.
- [ ] Chụp sample generation nếu cần.
- [ ] Lưu screenshot `02-sft-loss.png`.

Kiểm tra nhanh:

```bash
!ls -lh adapters/sft-mini
!cat adapters/sft-mini/adapter_config.json | head
```

### Phase 3 — Chạy NB2 preference data

- [ ] Chạy section NB2.
- [ ] Kiểm tra `train.parquet` tồn tại.
- [ ] Kiểm tra output in 3 examples có token counts.

Kiểm tra nhanh:

```bash
!ls -lh data/pref
```

### Phase 4 — Chạy NB3 DPO train

- [ ] Restart runtime nếu Colab memory đã bị đầy sau NB1/NB2.
- [ ] Nếu restart, mount/cd lại repo và đảm bảo artifacts còn đó.
- [ ] Chạy section NB3.
- [ ] Kiểm tra DPO adapter được lưu.
- [ ] Chụp reward curves.
- [ ] Lưu screenshot `03-dpo-reward-curves.png`.

Kiểm tra nhanh:

```bash
!ls -lh adapters/dpo
!cat adapters/dpo/dpo_metrics.json
```

Nếu OOM:

1. Restart runtime.
2. Chạy lại từ NB3 với artifacts đã lưu.
3. Đảm bảo tier vẫn là T4.
4. Không mở song song nhiều notebook.

### Phase 5 — Chạy NB4 compare and eval

- [ ] Chạy section NB4.
- [ ] Nếu không có API key, dùng manual rubric mode.
- [ ] Chụp bảng side-by-side.
- [ ] Chụp judge output hoặc manual rubric.
- [ ] Ghi lại win/loss/tie summary.

Artifacts:

```text
data/eval/side_by_side.jsonl
data/eval/judge_results.json
submission/screenshots/04-side-by-side-table.png
submission/screenshots/05-judge-output.png
```

### Phase 6 — Chạy NB5 merge + GGUF

- [ ] Chạy section NB5.
- [ ] Kiểm tra file GGUF nhỏ hơn 5 GB.
- [ ] Chạy smoke test.
- [ ] Chụp output smoke test tiếng Việt.
- [ ] Lưu screenshot `06-gguf-smoke.png`.

Kiểm tra nhanh:

```bash
!ls -lh gguf
```

### Phase 7 — Chạy NB6 benchmark

NB6 có thể lâu nhất sau DPO. Nếu Colab gần disconnect, ưu tiên lưu artifacts NB1–NB5 trước rồi chạy NB6 sau.

- [ ] Chạy section NB6.
- [ ] Kiểm tra `benchmark_results.json`.
- [ ] Chụp `07-benchmark-comparison.png`.
- [ ] Ghi lại điểm số để dùng trong `REFLECTION.md`.

Kiểm tra nhanh:

```bash
!ls -lh data/eval
!cat data/eval/benchmark_results.json
```

### Phase 8 — Verify trên Colab

Chạy:

```bash
!make verify
```

Nếu fail, sửa theo lỗi in ra. Những lỗi thường gặp:

- Thiếu screenshot.
- Thiếu `judge_results.json`.
- Thiếu `dpo_metrics.json`.
- Reflection còn placeholder.
- Thiếu benchmark chart.

## 7. Tải artifacts từ Colab về repo local

Sau khi chạy xong, cần đưa kết quả từ Colab về repo local này.

### Cách 1 — Tải zip artifacts

Trong Colab:

```bash
!zip -r day22-artifacts.zip adapters data gguf notebooks submission
```

Sau đó download `day22-artifacts.zip`, giải nén vào repo local:

```text
E:\LabAIThucChien\2A202600414-Nong-Trung-Kien-Day22-Track3
```

Ghi đè các thư mục tương ứng nếu cần:

```text
adapters/
data/
gguf/
notebooks/
submission/
```

Không ghi đè nhầm `PLAN_100_POINTS.md` và `instruction.md` nếu bạn đã chỉnh local.

### Cách 2 — Copy sang Google Drive

Trong Colab:

```bash
!mkdir -p /content/drive/MyDrive/day22-artifacts
!cp -r adapters data gguf notebooks submission /content/drive/MyDrive/day22-artifacts/
```

Sau đó tải từ Drive về local.

### Cách 3 — Commit trực tiếp từ Colab

Chỉ dùng nếu bạn quen git và đã cấu hình credential an toàn. Không commit API key hoặc `.env`.

```bash
!git status
!git add adapters data gguf notebooks submission
!git commit -m "add day22 colab artifacts"
!git push
```

Nếu không chắc credential, dùng Cách 1 hoặc Cách 2 an toàn hơn.

## 8. Viết `submission/REFLECTION.md` sau khi có số liệu

Sau khi có artifacts thật, điền reflection theo số liệu từ Colab.

### 8.1 Section 1 — Setup

Điền:

- GPU: `Google Colab Tesla T4 16GB`.
- CUDA/driver: lấy từ Colab output.
- Base model: `unsloth/Qwen2.5-3B-bnb-4bit`.
- SFT dataset slice: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated · 1000 samples · 1 epoch`.
- Preference dataset slice: `argilla/ultrafeedback-binarized-preferences-cleaned · 2000 pairs · 1 epoch`.
- `COMPUTE_TIER`: `T4`.
- Total cost: nếu dùng free Colab thì `$0`.

### 8.2 Section 3 — Reward curves analysis

Viết ít nhất 150 từ. Phải trả lời:

- `chosen_rewards` tăng, giảm, hay dao động?
- `rejected_rewards` tăng, giảm, hay dao động?
- Reward gap tăng do chosen tăng hay rejected giảm?
- Có dấu hiệu likelihood displacement không?
- DPO có làm model tốt hơn theo NB4/NB6 không?

Khung viết:

```text
Trong reward curve, chosen_rewards ...
Trong khi đó, rejected_rewards ...
Reward gap tăng từ ... lên ...
Điều quan trọng là gap tăng này đến từ ... chứ không chỉ ...
Theo deck §3.4, nếu chosen giảm nhưng rejected giảm mạnh hơn thì có thể là likelihood displacement. Trong run của tôi, ...
Vì vậy tôi diễn giải DPO run này là ...
```

### 8.3 Section 6 — Personal reflection

Viết ít nhất 150 từ. Với bối cảnh của bạn, quyết định phù hợp để viết là:

> Tôi chọn Google Colab T4 thay vì chạy local vì máy local chỉ có Intel Iris Xe shared memory, không có NVIDIA CUDA GPU.

Các ý cần có:

- Alternative: cố chạy local.
- Vì sao không chọn local: không CUDA, shared memory, DPO cần policy/reference, Colab T4 có 16GB VRAM.
- Kết quả có đúng kỳ vọng không.
- Nếu làm lại sẽ chuẩn bị Drive/artifact backup sớm hơn.

### 8.4 Section 7 — Benchmark interpretation

Viết ít nhất 150 từ. Phải dùng số từ `benchmark_results.json`.

Phân tích theo 4 benchmark:

- IFEval: instruction-following tăng/giảm thế nào.
- GSM8K: reasoning/math tăng/giảm thế nào.
- MMLU: factual knowledge giữ ổn hay giảm.
- AlpacaEval-lite: preference win-rate có khớp với NB4 judge không.

Nếu GSM8K hoặc MMLU giảm, giải thích bằng alignment tax.

## 9. Screenshots cần chụp trên Colab

Tối thiểu 6 ảnh, nên có đủ 7 ảnh:

- [ ] `01-setup-gpu.png` — Colab GPU output, thấy Tesla T4 hoặc `torch.cuda.get_device_name()`.
- [ ] `02-sft-loss.png` — SFT loss curve.
- [ ] `03-dpo-reward-curves.png` — chosen/rejected rewards và reward gap.
- [ ] `04-side-by-side-table.png` — bảng 8 prompts × 2 outputs.
- [ ] `05-judge-output.png` hoặc `05-manual-rubric.png` — judge verdict hoặc manual rubric.
- [ ] `06-gguf-smoke.png` — llama.cpp load Q4_K_M GGUF và sinh output tiếng Việt.
- [ ] `07-benchmark-comparison.png` — benchmark chart có delta annotations.

Nguyên tắc screenshot:

- Crop gọn vào dữ liệu.
- Không để lộ API key.
- Ảnh phải đọc được chữ.
- Nếu bảng dài, có thể chụp `04a` và `04b`.

## 10. Chạy verify và sửa lỗi

Sau khi tải artifacts về local hoặc ngay trên Colab, chạy:

```bash
make verify
```

Nếu local không đủ dependency để chạy verify, chạy trên Colab:

```bash
!make verify
```

`make verify` kiểm tra:

- Notebook source tồn tại.
- SFT adapter tồn tại.
- DPO adapter tồn tại.
- DPO metrics tồn tại.
- Preference parquet tồn tại.
- Side-by-side eval tồn tại.
- Judge results tồn tại.
- GGUF tồn tại và không quá 5 GB.
- Benchmark results tồn tại.
- `07-benchmark-comparison.png` tồn tại.
- Reflection không còn quá nhiều placeholder.
- Screenshot folder có ít nhất 6 ảnh.

Nếu verify pass nhưng reflection còn viết sơ sài, vẫn có thể mất điểm. Cần tự đối chiếu rubric.

## 11. Rủi ro Colab và cách xử lý

### Colab không cấp T4

Nếu Colab không cấp GPU hoặc cấp CPU:

- Runtime → Change runtime type → T4 GPU.
- Factory reset runtime.
- Đợi một lúc rồi thử lại.
- Dùng tài khoản Google khác nếu bị giới hạn quota.

### Colab disconnect giữa chừng

Cách giảm rủi ro:

- Sau NB1, NB3, NB5, copy artifacts sang Drive.
- Không để tab ngủ quá lâu.
- Chạy từng make target để biết đã xong đến đâu.

### OOM ở NB3 DPO

Cách xử lý:

- Restart runtime.
- Chỉ chạy lại từ NB3 nếu artifacts NB1/NB2 còn đó.
- Đảm bảo dùng T4 tier, không BigGPU tier.
- Không tăng `MAX_LEN`.
- Không mở thêm model khác trong cùng runtime.

### NB6 benchmark quá lâu

Cách xử lý:

- Chạy sau cùng.
- Đảm bảo đã backup artifacts trước NB6.
- Nếu timeout, chạy lại NB6 riêng.
- Nếu một benchmark fail, lưu lỗi và xem có output partial không; nhưng để lấy 100 điểm cần đủ `benchmark_results.json` và chart.

### Không có API key cho judge

Không sao. Dùng manual rubric mode:

- Ghi winner cho 8 prompts.
- Chụp `05-manual-rubric.png`.
- Viết rõ judge used: `manual rubric` trong `REFLECTION.md`.

### Tên screenshot hyphen vs underscore

`rubric.md` có một số tên underscore, còn screenshot README dùng hyphen. Cách an toàn:

- Dùng tên theo `submission/screenshots/README.md`.
- Trong `REFLECTION.md`, link rõ ảnh nào tương ứng tiêu chí nào.

## 12. Bonus tùy chọn

Không cần bonus để đạt 100 điểm core. Nếu còn thời gian trên Colab:

- [ ] β-sweep mini-experiment: +6.
- [ ] Push DPO adapter lên HuggingFace Hub: +5.
- [ ] Publish GGUF release với Q4_K_M và Q5_K_M: +3.
- [ ] MMLU full coverage: +3.
- [ ] Public W&B run link: +2.
- [ ] Cross-judge comparison: +4.

Với Colab free, ưu tiên core 100 trước. Bonus chỉ làm nếu core đã pass verify và còn runtime.

## 13. Definition of Done

Bài được coi là sẵn sàng submit khi:

- [ ] Colab T4 đã chạy đủ NB1 → NB6.
- [ ] 6 notebook đã chạy và output cells được preserve.
- [ ] `adapters/sft-mini/` đầy đủ.
- [ ] `adapters/dpo/` đầy đủ.
- [ ] `data/pref/train.parquet` tồn tại.
- [ ] `data/eval/side_by_side.jsonl` tồn tại.
- [ ] `data/eval/judge_results.json` tồn tại.
- [ ] `data/eval/benchmark_results.json` tồn tại.
- [ ] `gguf/` có Q4_K_M GGUF nhỏ hơn 5 GB.
- [ ] Screenshot evidence đầy đủ.
- [ ] `submission/REFLECTION.md` có số liệu thật và không còn placeholder.
- [ ] §3, §6, §7 đủ ít nhất 150 từ mỗi section.
- [ ] `make verify` pass trên Colab hoặc local.
- [ ] Artifacts từ Colab đã được đưa về repo local.
- [ ] Repo public.
- [ ] LMS nhận đúng public GitHub URL.

## 14. Thứ tự ưu tiên nếu gần deadline

Nếu gần deadline, làm theo thứ tự ưu tiên này:

1. Chạy NB1, NB2, NB3 để có SFT/DPO artifacts và reward curves.
2. Chạy NB4 để có side-by-side và win/loss/tie.
3. Chạy NB5 để có GGUF smoke.
4. Chạy NB6 để có benchmark JSON/chart.
5. Điền reflection thật kỹ §3, §6, §7.
6. Chạy `make verify`.
7. Push public repo.

Không nên dành thời gian cho bonus nếu core artifacts hoặc reflection chưa xong.
