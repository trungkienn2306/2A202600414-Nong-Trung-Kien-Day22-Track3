# Instruction — Day 22 DPO/ORPO Alignment Lab

Tài liệu này giải thích bài lab Day 22 làm gì, vì sao cần làm, pipeline chạy như thế nào, artifact nào được chấm, và các thuật ngữ quan trọng xuất hiện trong lab.

## 1. Bài lab hôm nay làm gì?

Day 22 là lab về **alignment** cho mô hình ngôn ngữ lớn. Mục tiêu là đi từ một model đã được fine-tune bằng SFT sang một model được align bằng preference learning, cụ thể là DPO.

Pipeline tổng quát:

```text
Base model
  → SFT-mini adapter
  → Preference dataset
  → DPO adapter
  → Compare SFT-only vs SFT+DPO
  → Merge + GGUF quantization
  → llama.cpp smoke test
  → Benchmark + reflection
```

Nói đơn giản: lab này kiểm tra xem sau khi model học từ các cặp câu trả lời “tốt hơn” và “kém hơn”, model có trả lời hữu ích hơn, an toàn hơn, và deploy được không.

## 2. Vì sao SFT chưa đủ?

SFT, viết tắt của Supervised Fine-Tuning, dạy model bắt chước câu trả lời mẫu. Cách này tốt để model học format, ngôn ngữ, phong cách, hoặc domain cơ bản.

Nhưng SFT có hạn chế:

- Nó chỉ học “đáp án mẫu”, không học trực tiếp sự so sánh giữa đáp án tốt và đáp án xấu.
- Nếu dataset có câu trả lời trung bình, model sẽ bắt chước cả những điểm chưa tốt.
- SFT không trực tiếp tối ưu helpfulness, safety, hoặc preference của người dùng.
- Khi gặp prompt nhạy cảm, model SFT có thể vẫn trả lời quá trực tiếp thay vì từ chối an toàn.

Vì vậy lab này thêm bước DPO để model học từ preference pairs: cùng một prompt, một response được chọn là tốt hơn, một response bị đánh giá là kém hơn.

## 3. Pipeline từng notebook

### 3.1 Notebook 01 — SFT-mini

File source:

```text
notebooks/01_sft_mini.py
```

Mục tiêu:

- Load base model theo tier.
- Gắn LoRA adapter.
- Train SFT-mini trên slice Vietnamese Alpaca.
- Lưu adapter vào `adapters/sft-mini/`.
- In sample generation để kiểm tra model hoạt động.

Input chính:

- Base model:
  - T4: `unsloth/Qwen2.5-3B-bnb-4bit`
  - BigGPU: `unsloth/Qwen2.5-7B-bnb-4bit`
- Dataset: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`

Output chính:

```text
adapters/sft-mini/adapter_config.json
notebooks/01_sft_mini.ipynb
submission/screenshots/02-sft-loss.png
```

Rubric chấm:

- Adapter config có `r=16`, `lora_alpha=32`.
- Loss curve giảm.
- Có sample generation.

### 3.2 Notebook 02 — Preference data

File source:

```text
notebooks/02_preference_data.py
```

Mục tiêu:

- Load preference dataset.
- Format dữ liệu thành 3 cột: `prompt`, `chosen`, `rejected`.
- Lưu Parquet để NB3 dùng train DPO.
- In 3 examples với token counts.

Input chính:

- Dataset: `argilla/ultrafeedback-binarized-preferences-cleaned`
- Tokenizer từ `adapters/sft-mini/`

Output chính:

```text
data/pref/train.parquet
data/pref/eval.parquet
notebooks/02_preference_data.ipynb
```

Rubric chấm:

- Parquet có đúng 3 cột `prompt`, `chosen`, `rejected`.
- 3 examples có token counts.
- `chosen` khác `rejected`.

### 3.3 Notebook 03 — DPO train

File source:

```text
notebooks/03_dpo_train.py
```

Mục tiêu:

- Load SFT-mini adapter.
- Train DPO adapter bằng TRL `DPOTrainer`.
- Dùng DPO hyperparameters: `beta=0.1`, `lr=5e-7`.
- Plot reward curves.
- Lưu adapter vào `adapters/dpo/`.

Input chính:

```text
adapters/sft-mini/
data/pref/train.parquet
```

Output chính:

```text
adapters/dpo/adapter_config.json
adapters/dpo/dpo_metrics.json
notebooks/03_dpo_train.ipynb
submission/screenshots/03-dpo-reward-curves.png
```

Rubric chấm:

- DPO adapter tồn tại.
- Reward gap tăng.
- Plot có cả `chosen_rewards` và `rejected_rewards`.
- Reflection diễn giải riêng hai đường reward này.

Đây là notebook quan trọng nhất vì chiếm 25/100 điểm.

### 3.4 Notebook 04 — Compare and eval

File source:

```text
notebooks/04_compare_and_eval.py
```

Mục tiêu:

- Tạo 8 fixed prompts.
- Generate response từ SFT-only.
- Generate response từ SFT+DPO.
- Render bảng side-by-side.
- Chạy judge tự động nếu có API key, hoặc dùng manual rubric.
- Lưu win/loss/tie summary.

Prompt mix:

- 4 helpfulness prompts.
- 4 safety prompts.

Output chính:

```text
data/eval/prompts.json
data/eval/side_by_side.jsonl
data/eval/judge_results.json
notebooks/04_compare_and_eval.ipynb
submission/screenshots/04-side-by-side-table.png
submission/screenshots/05-judge-output.png
```

Rubric chấm:

- Có bảng 8 prompts × 2 outputs.
- Có summary win/loss/tie.
- Có helpfulness/safety mix.

### 3.5 Notebook 05 — Merge, deploy, GGUF

File source:

```text
notebooks/05_merge_deploy_gguf.py
```

Mục tiêu:

- Load base model.
- Load SFT adapter.
- Apply DPO adapter.
- Merge adapter vào model weights.
- Quantize sang GGUF Q4_K_M.
- Smoke test bằng `llama-cpp-python`.

Output chính:

```text
adapters/merged-fp16/
gguf/lab22-dpo-Q4_K_M.gguf
notebooks/05_merge_deploy_gguf.ipynb
submission/screenshots/06-gguf-smoke.png
```

Rubric chấm:

- Có GGUF file nhỏ hơn 5 GB.
- Smoke test sinh output tiếng Việt coherent.
- Screenshot phải thể hiện filename Q4_K_M và generated tokens.

### 3.6 Notebook 06 — Benchmark

File source:

```text
notebooks/06_benchmark.py
```

Mục tiêu:

- Chạy benchmark trên SFT-only và SFT+DPO.
- So sánh điểm số trước/sau DPO.
- Plot chart có delta annotations.
- Dùng kết quả để phân tích alignment tax trong reflection.

Benchmark gồm:

- IFEval.
- GSM8K.
- MMLU.
- AlpacaEval-lite.

Output chính:

```text
data/eval/benchmark_results.json
notebooks/06_benchmark.ipynb
submission/screenshots/07-benchmark-comparison.png
```

Rubric chấm:

- JSON có 4 benchmark × 2 model scores.
- Chart có 4 nhóm benchmark và delta annotations.
- Reflection §7 giải thích benchmark nào tăng/giảm.

## 4. Các lệnh quan trọng

### Setup

```bash
bash setup-laptop.sh
```

### Smoke check

```bash
make smoke
```

### Chạy từng bước

```bash
make sft
make data
make dpo
make eval
make deploy
make bench
```

### Chạy toàn bộ pipeline

```bash
make pipeline
```

### Verify trước submit

```bash
make verify
```

### Mở Jupyter Lab

```bash
make lab
```

### Xóa artifact để chạy lại

```bash
make clean
```

Cẩn thận với `make clean` vì nó xóa adapters, data preference, GGUF và notebook ipynb đã convert.

## 5. Cách đọc kết quả

### 5.1 Cách đọc SFT loss curve

SFT loss curve cho biết model đang học bắt chước dữ liệu SFT tốt hơn hay không. Trong lab này, rubric yêu cầu loss giảm đơn điệu qua 1 epoch.

Nếu loss giảm:

- SFT training hoạt động.
- Adapter SFT-mini có tín hiệu học.
- Có thể dùng adapter này làm initial policy cho DPO.

Nếu loss không giảm:

- Có thể learning rate không phù hợp.
- Dataset format có lỗi.
- Model/tokenizer có vấn đề.
- Runtime hoặc gradient accumulation chưa đúng.

### 5.2 Cách đọc chosen/rejected rewards

Trong DPO, mỗi training example có:

- `prompt`: câu hỏi hoặc instruction.
- `chosen`: câu trả lời được prefer.
- `rejected`: câu trả lời bị đánh giá kém hơn.

Reward curves thường gồm:

- `chosen_rewards`: reward model/DPO objective gán cho chosen responses.
- `rejected_rewards`: reward model/DPO objective gán cho rejected responses.
- Reward gap: `chosen_rewards - rejected_rewards`.

Kết quả tốt nhất thường là:

- `chosen_rewards` tăng hoặc giữ ổn định.
- `rejected_rewards` giảm.
- Reward gap tăng.

Nhưng cần cẩn thận: reward gap tăng không tự động nghĩa là model tốt hơn. Nếu cả chosen và rejected đều giảm, nhưng rejected giảm nhanh hơn, gap vẫn tăng. Đây có thể là dấu hiệu của likelihood displacement.

### 5.3 Cách đọc side-by-side comparison

Bảng side-by-side so sánh model trước và sau DPO.

Khi đọc, cần xem:

- DPO có trả lời đúng yêu cầu hơn không?
- DPO có ngắn gọn, rõ ràng hơn không?
- DPO có từ chối các prompt nguy hiểm tốt hơn không?
- DPO có quá từ chối các prompt bình thường không?
- DPO có hallucination nhiều hơn hay ít hơn không?

Với safety prompts, response tốt thường là:

- Không cung cấp hướng dẫn gây hại.
- Từ chối lịch sự.
- Có thể đưa alternative an toàn.
- Với self-harm prompt, nên khuyến khích tìm hỗ trợ, nói chuyện với người đáng tin cậy, hoặc hotline nếu có.

### 5.4 Cách đọc benchmark chart

Benchmark chart so sánh điểm SFT-only và SFT+DPO trên 4 benchmark.

Cần đọc theo từng benchmark:

- IFEval: instruction-following.
- GSM8K: toán cấp tiểu học/trung học dạng word problems.
- MMLU: kiến thức rộng qua multiple-choice questions.
- AlpacaEval-lite: preference-style judge benchmark.

Nếu DPO tốt, có thể thấy:

- IFEval tăng vì model follow instruction tốt hơn.
- AlpacaEval-lite tăng vì model được judge prefer hơn.
- GSM8K hoặc MMLU có thể giảm nhẹ do alignment tax.

Nếu tất cả đều giảm mạnh, cần nghi ngờ training chưa ổn, over-alignment, data mismatch, hoặc output format bị ảnh hưởng.

## 6. Checklist artifact để nộp bài

### Adapter và data

- [ ] `adapters/sft-mini/adapter_config.json`
- [ ] `adapters/dpo/adapter_config.json`
- [ ] `adapters/dpo/dpo_metrics.json`
- [ ] `data/pref/train.parquet`
- [ ] `data/eval/side_by_side.jsonl`
- [ ] `data/eval/judge_results.json`
- [ ] `data/eval/benchmark_results.json`
- [ ] `gguf/lab22-dpo-Q4_K_M.gguf`

### Screenshots

- [ ] `01-setup-gpu.png`
- [ ] `02-sft-loss.png`
- [ ] `03-dpo-reward-curves.png`
- [ ] `04-side-by-side-table.png`
- [ ] `05-judge-output.png` hoặc `05-manual-rubric.png`
- [ ] `06-gguf-smoke.png`
- [ ] `07-benchmark-comparison.png`

### Reflection

- [ ] Có setup info.
- [ ] Có DPO experiment results.
- [ ] Có reward curves analysis ≥150 từ.
- [ ] Có qualitative comparison ≥8 examples.
- [ ] Có win/loss/tie summary.
- [ ] Có β trade-off hoặc hypothesis nếu không chạy sweep.
- [ ] Có personal reflection ≥150 từ.
- [ ] Có benchmark interpretation ≥150 từ.
- [ ] Không còn placeholder.

## 7. Glossary thuật ngữ

### Alignment

Alignment là quá trình làm cho model hành xử phù hợp hơn với ý định, giá trị, tiêu chí an toàn, và preference của con người. Một model aligned hơn không chỉ trả lời đúng hơn, mà còn biết trả lời hữu ích, tránh gây hại, và tuân thủ instruction tốt hơn.

### SFT — Supervised Fine-Tuning

SFT là fine-tuning có giám sát. Model học từ các cặp input-output mẫu. Trong lab này, NB1 train SFT-mini adapter để tạo checkpoint nền trước khi train DPO.

### Preference learning

Preference learning là cách train model từ dữ liệu so sánh. Thay vì chỉ nói “đây là đáp án đúng”, dataset nói “với prompt này, response A tốt hơn response B”. DPO là một phương pháp preference learning.

### DPO — Direct Preference Optimization

DPO là phương pháp tối ưu model trực tiếp từ preference pairs mà không cần train reward model riêng như RLHF truyền thống. DPO học để tăng xác suất của `chosen` response so với `rejected` response, có regularization với reference model thông qua tham số β.

### ORPO — Odds Ratio Preference Optimization

ORPO là một phương pháp preference optimization khác. ORPO kết hợp supervised fine-tuning objective với odds-ratio preference objective trong một quá trình train. Lab này có tên DPO/ORPO vì bài học nói về nhóm phương pháp preference alignment, nhưng core pipeline trong repo tập trung vào DPO.

### RLHF — Reinforcement Learning from Human Feedback

RLHF là pipeline align model bằng feedback của con người. Cách truyền thống thường gồm SFT, train reward model, rồi dùng reinforcement learning như PPO để tối ưu policy. DPO đơn giản hóa pipeline này bằng cách bỏ reward model riêng và tối ưu trực tiếp từ preference pairs.

### LoRA — Low-Rank Adaptation

LoRA là kỹ thuật fine-tuning nhẹ. Thay vì cập nhật toàn bộ model weights, LoRA thêm các ma trận low-rank nhỏ vào một số layer như `q_proj`, `k_proj`, `v_proj`, `o_proj`. Điều này giảm VRAM, giảm dung lượng checkpoint, và giúp train model lớn trên GPU nhỏ hơn.

### QLoRA — Quantized LoRA

QLoRA là LoRA trên base model đã quantize, thường là 4-bit. Base model được giữ ở precision thấp để tiết kiệm VRAM, còn LoRA adapter vẫn học các tham số nhỏ. Lab này dùng base model dạng bnb-4bit.

### PEFT — Parameter-Efficient Fine-Tuning

PEFT là nhóm kỹ thuật fine-tuning chỉ cập nhật một phần nhỏ tham số, ví dụ LoRA. PEFT giúp fine-tune model lớn với chi phí thấp hơn full fine-tuning.

### Adapter

Adapter là phần weights nhỏ được train thêm trên base model. Trong lab này có SFT adapter ở `adapters/sft-mini/` và DPO adapter ở `adapters/dpo/`. Adapter có thể được load lên base model để thay đổi hành vi mà không cần lưu toàn bộ model.

### Base model

Base model là model gốc trước khi fine-tune trong lab. Ví dụ T4 tier dùng `unsloth/Qwen2.5-3B-bnb-4bit`; BigGPU tier dùng `unsloth/Qwen2.5-7B-bnb-4bit`.

### Policy model

Policy model là model đang được train hoặc tối ưu. Trong DPO, policy model bắt đầu từ SFT model và được cập nhật để prefer chosen responses hơn rejected responses.

### Reference model

Reference model là model tham chiếu, thường là SFT model ban đầu, được giữ cố định. DPO so sánh policy model với reference model để tránh policy drift quá mạnh.

### Policy drift

Policy drift là hiện tượng model sau khi tối ưu đi quá xa khỏi model ban đầu. Nếu drift quá mạnh, model có thể mất kiến thức, trả lời lạ, hoặc overfit preference data.

### Prompt

Prompt là đầu vào người dùng đưa cho model. Trong preference dataset, cùng một prompt sẽ có một chosen response và một rejected response.

### Chosen response

Chosen response là câu trả lời được đánh giá tốt hơn trong một preference pair. DPO cố gắng làm model tăng xác suất sinh ra response kiểu này.

### Rejected response

Rejected response là câu trả lời bị đánh giá kém hơn trong một preference pair. DPO cố gắng làm model giảm xác suất sinh ra response kiểu này.

### Reward

Reward là tín hiệu điểm số biểu diễn response được prefer đến mức nào. Trong DPO, reward không nhất thiết đến từ reward model riêng, mà được suy ra từ log-probability ratio giữa policy và reference.

### `chosen_rewards`

`chosen_rewards` là reward trajectory cho chosen responses trong quá trình DPO train. Nếu đường này tăng, model đang học đánh giá hoặc xác suất hóa chosen responses tốt hơn.

### `rejected_rewards`

`rejected_rewards` là reward trajectory cho rejected responses. Nếu đường này giảm, model đang học tránh các response kém hơn.

### Reward gap

Reward gap là hiệu giữa chosen reward và rejected reward:

```text
reward_gap = chosen_rewards - rejected_rewards
```

Reward gap tăng thường là tín hiệu DPO đang phân biệt tốt hơn giữa response tốt và xấu. Tuy nhiên cần xem riêng hai đường chosen/rejected để tránh hiểu nhầm.

### β — Beta trong DPO

β điều khiển mức độ conservative của DPO so với reference model. β nhỏ thường cho phép policy thay đổi mạnh hơn. β lớn giữ policy gần reference model hơn. Trong lab, default là `beta=0.1`.

### KL divergence

KL divergence đo độ khác biệt giữa hai phân phối xác suất. Trong alignment, nó thường được dùng để đo policy model đã lệch khỏi reference model bao nhiêu. KL quá cao có thể báo hiệu model drift quá mạnh.

### Likelihood displacement

Likelihood displacement là failure mode trong DPO. Reward gap có thể tăng không phải vì model thích chosen hơn một cách lành mạnh, mà vì likelihood của rejected bị đẩy xuống mạnh, hoặc cả chosen/rejected cùng giảm nhưng rejected giảm nhanh hơn. Vì vậy rubric yêu cầu plot và diễn giải riêng chosen/rejected trajectories.

### Length hacking

Length hacking là hiện tượng model học rằng câu trả lời dài hơn hoặc ngắn hơn sẽ được prefer, thay vì học chất lượng thật. Nếu DPO làm output length thay đổi bất thường, cần kiểm tra xem model có đang tối ưu độ dài thay vì nội dung không.

### Alignment tax

Alignment tax là trade-off khi model được align tốt hơn về instruction-following, helpfulness, hoặc safety, nhưng một số năng lực khác như reasoning, math, hoặc factual knowledge giảm nhẹ. Trong lab, GSM8K hoặc MMLU giảm sau DPO có thể là alignment tax.

### Helpfulness

Helpfulness là mức độ câu trả lời hữu ích, đúng yêu cầu, rõ ràng, có cấu trúc và giải quyết được vấn đề của người dùng.

### Safety

Safety là khả năng model tránh đưa ra hướng dẫn gây hại, ví dụ bạo lực, tự hại, lừa đảo, chất nổ, hoặc nội dung nguy hiểm. Safety tốt không chỉ là từ chối, mà còn nên đưa hướng an toàn nếu phù hợp.

### Judge model

Judge model là model khác được dùng để đánh giá output của hai model cần so sánh. Ví dụ có thể dùng GPT-4o-mini hoặc Claude Haiku để chọn response tốt hơn giữa SFT-only và SFT+DPO.

### Manual rubric

Manual rubric là cách tự chấm bằng tiêu chí rõ ràng khi không có API key cho judge model. Rubric thủ công vẫn được chấp nhận nếu có evidence và win/loss/tie summary.

### Win/loss/tie summary

Win/loss/tie summary là thống kê DPO thắng, thua, hoặc hòa SFT-only trên 8 prompts. Ví dụ: DPO wins 5/8, ties 2/8, loses 1/8.

### GGUF

GGUF là định dạng model file phổ biến trong llama.cpp ecosystem. GGUF phù hợp để chạy inference local, nhất là sau khi quantize.

### Quantization

Quantization là quá trình giảm precision của weights, ví dụ từ FP16 xuống 4-bit. Mục tiêu là giảm dung lượng file, giảm RAM/VRAM khi inference, và tăng tốc trong một số môi trường.

### Q4_K_M

Q4_K_M là một kiểu quantization 4-bit trong llama.cpp/GGUF. Nó thường là điểm cân bằng tốt giữa dung lượng nhỏ và chất lượng giữ được. Rubric yêu cầu GGUF nhỏ hơn 5 GB, Q4_K_M giúp đạt điều đó.

### llama.cpp

llama.cpp là framework inference C/C++ cho LLM, nổi tiếng vì chạy được model quantized trên CPU, GPU, Metal, CUDA, và nhiều môi trường local khác.

### llama-cpp-python

llama-cpp-python là binding Python cho llama.cpp. Trong lab, NB5 dùng nó để load GGUF và chạy smoke test.

### Smoke test

Smoke test là kiểm tra nhanh xem artifact có chạy được không. Ở NB5, smoke test load GGUF và sinh một câu trả lời tiếng Việt để chứng minh model deploy được.

### Benchmark

Benchmark là bài đánh giá định lượng. Lab dùng benchmark để so sánh SFT-only và SFT+DPO bằng số, không chỉ nhìn vài ví dụ qualitative.

### IFEval

IFEval là benchmark đo khả năng follow instruction chính xác, ví dụ yêu cầu trả lời theo format cụ thể. DPO thường có cơ hội cải thiện IFEval nếu alignment tốt.

### GSM8K

GSM8K là benchmark toán dạng word problems. Nó thường dùng để đo reasoning toán học. Sau alignment, GSM8K có thể giảm nhẹ do alignment tax.

### MMLU

MMLU là benchmark trắc nghiệm kiến thức rộng trên nhiều lĩnh vực. Nếu MMLU giảm mạnh sau DPO, có thể model bị mất năng lực factual hoặc over-aligned.

### AlpacaEval-lite

AlpacaEval-lite là benchmark preference-style, dùng judge để so sánh response. Nó gần với mục tiêu DPO hơn các benchmark static vì cũng dựa trên preference.

### Unsloth

Unsloth là thư viện giúp fine-tune LLM nhanh và tiết kiệm VRAM hơn, đặc biệt với LoRA/QLoRA. Lab dùng Unsloth để train SFT và DPO trên GPU phổ thông như T4.

### TRL

TRL là thư viện của Hugging Face cho reinforcement learning và preference optimization với language models. Lab dùng `SFTTrainer`, `DPOTrainer`, và `DPOConfig` từ TRL.

### bitsandbytes

bitsandbytes là thư viện hỗ trợ quantization và optimizer tiết kiệm bộ nhớ, ví dụ 4-bit loading và 8-bit AdamW.

### vLLM

vLLM là inference/serving engine hiệu năng cao cho LLM. Trong lab này vLLM là phần BigGPU hoặc production-style serving, không phải đường mặc định cho T4.

### Jupytext

Jupytext đồng bộ notebook `.ipynb` với file source `.py`. Repo này lưu notebook source ở dạng `.py` để dễ review, rồi convert sang `.ipynb` khi chạy.

### Parquet

Parquet là định dạng file dữ liệu dạng cột, phổ biến cho dataset ML. NB2 lưu preference dataset thành `data/pref/train.parquet`.

### VRAM

VRAM là bộ nhớ GPU. DPO cần nhiều VRAM hơn SFT vì phải giữ policy/reference model hoặc adapter state liên quan trong memory.

### CUDA

CUDA là nền tảng tính toán GPU của NVIDIA. Lab cần GPU CUDA để train model hiệu quả.

### T4 tier

T4 tier là cấu hình phù hợp Free Colab T4 hoặc laptop GPU khoảng 12–16 GB VRAM. Nó dùng Qwen2.5-3B 4-bit, sequence length nhỏ hơn, và dataset slice nhỏ hơn.

### BigGPU tier

BigGPU tier là cấu hình cho GPU mạnh hơn như A100, L4, H100. Nó có thể dùng Qwen2.5-7B 4-bit và dataset slice lớn hơn.

### API key

API key là secret dùng để gọi dịch vụ bên ngoài như OpenAI hoặc Anthropic judge. Không bao giờ để API key xuất hiện trong screenshot hoặc commit vào repo.

### HuggingFace Hub

HuggingFace Hub là nơi publish model, adapter, dataset, và model card. Trong rubric, push DPO adapter lên Hub là bonus.

### W&B — Weights & Biases

Weights & Biases là công cụ tracking experiment. Trong rubric, public W&B run link là bonus nếu có training curves rõ ràng.

## 8. Cách viết reflection để không mất điểm

### Section 3 — Reward curves analysis

Không viết chung chung kiểu “reward gap tăng nên DPO tốt”. Nên viết theo cấu trúc:

1. Mô tả chosen reward.
2. Mô tả rejected reward.
3. Mô tả reward gap.
4. Giải thích gap tăng lành mạnh hay có dấu hiệu likelihood displacement.
5. Kết luận DPO có đạt mục tiêu không.

### Section 6 — Personal reflection

Chọn một quyết định thật, ví dụ chọn T4 thay vì BigGPU. Viết:

1. Alternative là gì.
2. Vì sao chọn phương án hiện tại.
3. Kết quả có đúng kỳ vọng không.
4. Nếu làm lại sẽ đổi gì.

### Section 7 — Benchmark interpretation

Nên có bảng số liệu từ `benchmark_results.json`, rồi phân tích:

1. Benchmark nào tăng nhiều nhất.
2. Benchmark nào giảm.
3. Có alignment tax không.
4. NB6 có khớp với NB4 qualitative judge không.
5. Kết quả nói gì về chất lượng DPO.

## 9. Một cách hiểu ngắn gọn toàn lab

Nếu phải tóm tắt lab trong một câu:

> Lab này dạy cách biến một model SFT thành model được align bằng preference data, kiểm tra alignment bằng reward curves, judge, benchmark, rồi export thành GGUF có thể chạy local.

Nếu phải nhớ 3 điều quan trọng nhất:

1. DPO không chỉ cần reward gap tăng; phải xem riêng chosen và rejected rewards.
2. Alignment tốt có thể đi kèm alignment tax trên một số benchmark reasoning hoặc knowledge.
3. Artifact chạy được, screenshot rõ, reflection có số liệu thật mới đủ chắc để lấy 100 điểm.
