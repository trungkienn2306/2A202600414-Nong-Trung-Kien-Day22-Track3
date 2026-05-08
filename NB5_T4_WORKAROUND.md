# NB5 T4 workaround — bỏ qua step 2 lỗi nhưng vẫn đạt rubric

## Kết luận ngắn

Theo `rubric.md`, phần NB5 chỉ chấm 2 thứ:

- `rubric.md:22` — có file `gguf/lab22-dpo-Q4_K_M.gguf`
- `rubric.md:23` — có screenshot smoke test `06_gguf_smoke.png` với output tiếng Việt coherent

**Rubric không chấm việc phải lưu `merged-fp16/` theo đúng cách hiện tại.**

Ngoài ra, code NB5 hiện tại còn có lỗi logic:

- `notebooks/05_merge_deploy_gguf.py:68-92` đang load `SFT_PATH` chứ không dùng `DPO_PATH`
- nghĩa là nếu merge chạy được thì cũng đang có nguy cơ merge ra **SFT-only**, không phải model DPO cuối cùng

Vì vậy, trên Colab T4, giải pháp an toàn nhất là:

1. **bỏ qua step 2 cũ**
2. **không dùng cell merge FP16 cũ**
3. **tạo cell mới để export GGUF trực tiếp từ `adapters/dpo/`**
4. smoke-test GGUF để lấy screenshot nộp rubric

---

## Bạn nên skip những cell nào trong NB5

Sau khi đã chạy thành công:

- **0. Setup**
- **1. Load DPO model + merge adapter**

thì:

### Hãy bỏ qua các cell cũ sau
- `## 2. Save merged FP16 weights`
- `## 3. Quantize to GGUF Q4_K_M`
- cell reload `merged-fp16`

Thay vào đó, tạo **cell mới** bên dưới.

---

## Cell mới A — export GGUF trực tiếp từ DPO adapter

> Dán cell này vào Colab sau step 1, rồi chạy nó.

```python
# NB5 workaround for T4: export GGUF directly from adapters/dpo
import gc
import torch
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel

DPO_PATH = REPO_ROOT / "adapters" / "dpo"
GGUF_DIR = REPO_ROOT / "gguf"
GGUF_DIR.mkdir(parents=True, exist_ok=True)

assert DPO_PATH.exists(), f"DPO adapter missing: {DPO_PATH}"

print(f"BASE_MODEL: {BASE_MODEL}")
print(f"DPO_PATH:   {DPO_PATH}")
print(f"GGUF_DIR:   {GGUF_DIR}")

# Clean up anything loaded by earlier cells
try:
    del model
except NameError:
    pass

gc.collect()
torch.cuda.empty_cache()

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the final DPO adapter
model = PeftModel.from_pretrained(model, str(DPO_PATH))
print("Loaded DPO adapter.")

# Some merge paths are sensitive to tied embeddings
if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
    model.config.tie_word_embeddings = False

# Export directly to GGUF
model.save_pretrained_gguf(
    str(GGUF_DIR),
    tokenizer,
    quantization_method="q4_k_m",
)

print("Saved GGUF files:")
for p in sorted(GGUF_DIR.iterdir()):
    if p.suffix == ".gguf":
        print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
```

---

## Cell mới B — smoke test GGUF

> Dán cell này ngay sau cell A để test và lấy ảnh chụp nộp bài.

```python
from llama_cpp import Llama

# Find Q4_K_M GGUF
candidates = list(GGUF_DIR.glob("*Q4_K_M*.gguf")) + list(GGUF_DIR.glob("*q4_k_m*.gguf"))
assert candidates, f"No Q4_K_M GGUF found in {GGUF_DIR}"
gguf_path = candidates[0]
print(f"Using GGUF: {gguf_path.name}")

# Try GPU first; fall back to CPU if llama-cpp wheel is CPU-only
try:
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=MAX_LEN,
        n_gpu_layers=-1,
        chat_format="chatml",
        verbose=False,
    )
    print("Loaded GGUF with GPU offload.")
except Exception as exc:
    print(f"GPU load failed, falling back to CPU: {exc}")
    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=MAX_LEN,
        n_gpu_layers=0,
        chat_format="chatml",
        verbose=False,
    )
    print("Loaded GGUF on CPU.")

SMOKE_PROMPT = "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động."

response = llm.create_chat_completion(
    messages=[{"role": "user", "content": SMOKE_PROMPT}],
    max_tokens=200,
    temperature=0.0,
)

print(f"PROMPT:\n{SMOKE_PROMPT}\n")
print("RESPONSE:")
print(response["choices"][0]["message"]["content"])
print("\nUSAGE:")
print(response.get("usage", {}))
```

---

## Cell mới C — lưu deployment metadata

> Nếu cell smoke test chạy được, dán thêm cell này để đồng bộ artifact với repo.

```python
import json

deploy_meta = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "gguf_path": str(gguf_path),
    "gguf_size_mb": round(gguf_path.stat().st_size / 1e6, 1),
    "quantization": "q4_k_m",
    "smoke_prompt": SMOKE_PROMPT,
    "smoke_response": response["choices"][0]["message"]["content"],
}

out_path = REPO_ROOT / "data" / "eval" / "deploy_meta.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(deploy_meta, ensure_ascii=False, indent=2))
print(f"Saved {out_path}")
```

---

## Nếu `llama_cpp` load lỗi

Nếu lỗi import hoặc CUDA wheel của `llama-cpp-python`, tạo 1 cell cài lại:

```python
!CMAKE_ARGS="-DGGML_CUDA=on" pip install -U --force-reinstall --no-cache-dir llama-cpp-python
```

Sau đó restart runtime rồi chạy lại riêng các cell smoke test.

---

## Cách chụp ảnh để đủ rubric NB5

Bạn cần 1 ảnh `06_gguf_smoke.png` có đủ:

1. tên file GGUF đang load
2. prompt tiếng Việt
3. response tiếng Việt coherent
4. nếu được, hiện luôn usage/tokens

Ảnh này chính là bằng chứng cho `rubric.md:23`.

---

## Khuyến nghị thực tế

Trên T4, mục tiêu nên là:

- lấy được `Q4_K_M.gguf`
- chạy smoke test thành công
- chụp ảnh nộp rubric

Không cần cố giữ nguyên đường đi `merged-fp16 -> reload -> quantize` nếu chính bước đó đang fail, vì **rubric không chấm đường đi đó, chỉ chấm artifact cuối**.
