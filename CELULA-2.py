# ============================================================
# CÉLULA 2: VERIFICAÇÃO - Execute após reiniciar o runtime
# ============================================================
import torch
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

import unsloth
print(f"🦥 Unsloth: {unsloth.__version__}")

from unsloth import FastLanguageModel
print("✅ FastLanguageModel importado com sucesso!")

from transformers import AutoTokenizer, TrainingArguments, Trainer
print("✅ Transformers importado com sucesso!")

from peft import PeftModel, LoraConfig
print("✅ PEFT importado com sucesso!")

from datasets import load_dataset
print("✅ Datasets importado com sucesso!")

print("\n🎉 Tudo pronto! Execute a Célula 3 para iniciar o treinamento.")
