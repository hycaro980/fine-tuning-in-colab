#!/usr/bin/env python3
"""
Pipeline de Fine-Tuning com Unsloth + LoRA - Skskskd/bliz-ia-PT-BR
SEM quantização 4-bit | Otimizado com Unsloth
CORRIGIDO: Suporte para modelos com adaptadores pré-existentes
"""

import os
import sys
import json
import torch
import getpass
from typing import Optional

from unsloth import FastLanguageModel
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login, HfApi

# ============================================================
# CONFIGURAÇÕES GLOBAIS
# ============================================================
MODELO_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_INICIAL = "Skskskd/teste1"
DIR_BASE_MODELOS = "./modelo_treinado"
versao_atual = 0
modelo = None
tokenizer = None
historico_perdas = []

# ============================================================
# DETECÇÃO DE GPU
# ============================================================
def detectar_gpu():
    if not torch.cuda.is_available():
        print("⚠️ Nenhuma GPU detectada.")
        return False, "cpu"

    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🖥️ GPU: {gpu_name}")
    print(f"💾 VRAM: {vram:.1f} GB")

    gpu_mode = "limited_vram" if vram < 16 else "adequate_vram"
    print(f"📌 Modo: {gpu_mode}")
    return True, gpu_mode

# ============================================================
# VALIDAÇÃO
# ============================================================
def validar_modelo(modelo_id):
    print(f"\n🔍 Validando modelo: {modelo_id}...")
    try:
        AutoTokenizer.from_pretrained(modelo_id, trust_remote_code=True)
        print(f"✅ Modelo encontrado.")
        return True
    except Exception as e:
        print(f"❌ Modelo não encontrado: {e}")
        return False

def validar_dataset(dataset_id):
    print(f"\n🔍 Validando dataset: {dataset_id}...")
    try:
        ds = load_dataset(dataset_id, split="train[:5]")
        print(f"✅ Dataset encontrado. Colunas: {ds.column_names}")
        print(f"   Amostra: {ds[0]}")
        return True
    except Exception as e:
        print(f"❌ Dataset não encontrado: {e}")
        return False

# ============================================================
# CARREGAMENTO DO MODELO COM UNSLOTH (CORRIGIDO)
# ============================================================
def carregar_modelo(modelo_id, gpu_mode="adequate_vram", adaptadores_path=None):
    global modelo, tokenizer

    print(f"\n📦 Carregando modelo com Unsloth: {modelo_id}...")

    max_seq_length = 1024
    dtype = torch.bfloat16 if gpu_mode == "adequate_vram" else torch.float16

    # Carregar modelo base
    modelo, tokenizer = FastLanguageModel.from_pretrained(
        model_name=modelo_id,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
        trust_remote_code=True,
    )

    # VERIFICAR SE O MODELO JÁ TEM ADAPTADORES LoRA
    # Se tiver, precisamos mesclar para poder aplicar novos
    if hasattr(modelo, 'peft_config') or 'peft' in str(type(modelo)).lower():
        print(f"⚠️ Modelo já possui adaptadores LoRA pré-treinados.")
        print(f"🔄 Mesclando adaptadores ao modelo base...")

        # Mesclar adaptadores existentes
        modelo = modelo.merge_and_unload()
        print(f"✅ Adaptadores mesclados. Modelo base limpo para novo treinamento.")

    # Carregar adaptadores adicionais se especificado
    if adaptadores_path and os.path.exists(adaptadores_path):
        print(f"🔄 Carregando adaptadores adicionais de: {adaptadores_path}")
        modelo = PeftModel.from_pretrained(modelo, adaptadores_path)
        # Mesclar também esses adaptadores
        modelo = modelo.merge_and_unload()
        print("✅ Adaptadores adicionais mesclados.")

    print(f"✅ Modelo carregado | dtype={dtype} | seq_len={max_seq_length}")
    print(f"   Parâmetros: {modelo.num_parameters()/1e6:.1f}M")

    return modelo, tokenizer

# ============================================================
# CONFIGURAÇÃO LoRA COM UNSLOTH
# ============================================================
def configurar_lora(modelo):
    print(f"\n🔧 Configurando LoRA com Unsloth...")

    modelo_lora = FastLanguageModel.get_peft_model(
        modelo,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
    )

    params_treinaveis = sum(p.numel() for p in modelo_lora.parameters() if p.requires_grad)
    params_total = sum(p.numel() for p in modelo_lora.parameters())

    print(f"📊 LoRA configurado:")
    print(f"   r=16, alpha=32, dropout=0.05")
    print(f"   Gradient checkpointing: ON")
    print(f"   Params treináveis: {params_treinaveis:,} ({params_treinaveis/params_total*100:.2f}%)")

    return modelo_lora

# ============================================================
# PREPARAÇÃO DO DATASET
# ============================================================
def preparar_dataset(dataset_id, tok):
    print(f"\n📂 Carregando dataset: {dataset_id}...")

    dataset = load_dataset(dataset_id)
    train_data = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]

    print(f"   Exemplos: {len(train_data)}")
    print(f"   Colunas: {train_data.column_names}")

    eos = tok.eos_token

    def formatar(ex):
        inst = ex.get("instruction", "")
        inp = ex.get("input", "")
        out = ex.get("output", "")

        if inp and inp.strip():
            texto = f"### Instrução:\n{inst}\n\n### Entrada:\n{inp}\n\n### Resposta:\n{out}{eos}"
        else:
            texto = f"### Instrução:\n{inst}\n\n### Resposta:\n{out}{eos}"

        return {"text": texto}

    dataset_fmt = train_data.map(formatar, remove_columns=train_data.column_names)

    def tokenizar(ex):
        r = tok(ex["text"], truncation=True, max_length=1024, padding="max_length")
        r["labels"] = r["input_ids"].copy()
        return r

    dataset_tok = dataset_fmt.map(tokenizar, remove_columns=["text"], num_proc=2)

    print(f"✅ Dataset tokenizado: {len(dataset_tok)} exemplos")
    return dataset_tok

# ============================================================
# TREINAMENTO
# ============================================================
def treinar_modelo(modelo_lora, dataset_tok, output_dir, gpu_mode):
    global historico_perdas

    print(f"\n🏋️ Iniciando treinamento...")

    batch_size = 2 if gpu_mode == "limited_vram" else 4
    grad_accum = 8 if gpu_mode == "limited_vram" else 4

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=(gpu_mode == "adequate_vram"),
        fp16=(gpu_mode == "limited_vram"),
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        seed=42,
    )

    print(f"   Batch: {batch_size} | Grad accum: {grad_accum} | Effective: {batch_size * grad_accum}")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=modelo_lora,
        args=training_args,
        train_dataset=dataset_tok,
        data_collator=data_collator,
    )

    class LossLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss = logs["loss"]
                step = state.global_step
                historico_perdas.append({"step": step, "loss": loss})
                lr = logs.get("learning_rate", 0)
                print(f"   📉 Step {step:>5d} | Loss: {loss:.4f} | LR: {lr:.2e}")

        def on_train_end(self, args, state, control, **kwargs):
            print(f"\n📊 Resumo:")
            print(f"   Steps: {state.global_step}")
            if historico_perdas:
                print(f"   Loss inicial: {historico_perdas[0]['loss']:.4f}")
                print(f"   Loss final:   {historico_perdas[-1]['loss']:.4f}")

    trainer.add_callback(LossLogger())

    print(f"⚡ Treinando com Unsloth (até 2x mais rápido)...")
    resultado = trainer.train()

    print(f"\n✅ Treinamento concluído!")
    print(f"   Loss final: {resultado.training_loss:.4f}")
    print(f"   Tempo: {resultado.metrics['train_runtime']/60:.1f} min")

    return trainer

# ============================================================
# SALVAMENTO
# ============================================================
def salvar_adaptadores(trainer, output_dir):
    print(f"\n💾 Salvando em: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    config = {
        "framework": "unsloth",
        "model_base": MODELO_BASE,
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "quantization": "none",
    }
    with open(os.path.join(output_dir, "unsloth_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"   Arquivos: {os.listdir(output_dir)}")
    print(f"✅ Salvamento concluído!")

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def executar_treinamento(dataset_id, adaptadores_previos=None):
    global versao_atual, modelo, tokenizer

    versao_atual += 1
    output_dir = f"{DIR_BASE_MODELOS}_v{versao_atual}"

    print(f"\n{'='*60}")
    print(f"🚀 TREINAMENTO v{versao_atual} | Unsloth (sem 4-bit)")
    print(f"{'='*60}")
    print(f"   Modelo: {MODELO_BASE}")
    print(f"   Dataset: {dataset_id}")
    print(f"   Adaptadores prev: {adaptadores_previos or 'Nenhum'}")
    print(f"   Output: {output_dir}")
    print(f"{'='*60}")

    tem_gpu, gpu_mode = detectar_gpu()

    if not validar_modelo(MODELO_BASE):
        return False
    if not validar_dataset(dataset_id):
        return False

    modelo, tokenizer = carregar_modelo(MODELO_BASE, gpu_mode, adaptadores_previos)
    modelo_lora = configurar_lora(modelo)
    dataset_tok = preparar_dataset(dataset_id, tokenizer)
    trainer = treinar_modelo(modelo_lora, dataset_tok, output_dir, gpu_mode)
    salvar_adaptadores(trainer, output_dir)

    del trainer, modelo_lora, dataset_tok
    torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"🎉 TREINAMENTO v{versao_atual} CONCLUÍDO!")
    print(f"{'='*60}")
    return True

# ============================================================
# MODO DE TESTE
# ============================================================
def modo_teste():
    global modelo, tokenizer

    adaptadores_path = f"{DIR_BASE_MODELOS}_v{versao_atual}"
    if not os.path.exists(adaptadores_path):
        print(f"❌ Adaptadores não encontrados: {adaptadores_path}")
        return

    print(f"\n🔄 Carregando modelo para teste...")
    _, gpu_mode = detectar_gpu()
    modelo_teste, tok_teste = carregar_modelo(MODELO_BASE, gpu_mode, adaptadores_path)
    modelo_teste = FastLanguageModel.for_inference(modelo_teste)
    modelo_teste.eval()

    print(f"\n{'='*60}")
    print(f"💬 CHAT INTERATIVO (digite 'sair' para voltar)")
    print(f"{'='*60}\n")

    while True:
        prompt = input("👤 Você: ").strip()
        if prompt.lower() in ["sair", "exit", "quit"]:
            print("👋 Retornando ao menu...")
            break
        if not prompt:
            continue

        texto = f"### Instrução:\n{prompt}\n\n### Resposta:\n"
        inputs = tok_teste(texto, return_tensors="pt").to(modelo_teste.device)

        with torch.no_grad():
            outputs = modelo_teste.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=tok_teste.eos_token_id,
            )

        resp = tok_teste.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        print(f"\n🤖 Modelo: {resp}\n")
        print("-" * 40)

    del modelo_teste
    torch.cuda.empty_cache()

# ============================================================
# UPLOAD HUGGING FACE
# ============================================================
def upload_huggingface():
    adaptadores_path = f"{DIR_BASE_MODELOS}_v{versao_atual}"
    if not os.path.exists(adaptadores_path):
        print(f"❌ Adaptadores não encontrados: {adaptadores_path}")
        return

    print(f"\n{'='*60}")
    print(f"📤 UPLOAD PARA HUGGING FACE")
    print(f"{'='*60}")

    token = getpass.getpass("\n🔑 Token HF (https://huggingface.co/settings/tokens): ")
    if not token.strip():
        print("❌ Token vazio.")
        return

    try:
        login(token=token.strip())
        print("✅ Login OK!")
    except Exception as e:
        print(f"❌ Falha: {e}")
        return

    repo_nome = input("\n📁 Nome do repositório (ex: usuario/modelo-lora): ").strip()
    if not repo_nome:
        print("❌ Nome vazio.")
        return

    print(f"\n⬆️ Upload para: {repo_nome}...")
    try:
        api = HfApi()
        api.create_repo(repo_id=repo_nome, exist_ok=True)
        api.upload_folder(
            folder_path=adaptadores_path,
            repo_id=repo_nome,
            repo_type="model",
            commit_message=f"LoRA v{versao_atual} - Unsloth otimizado",
        )
        print(f"\n✅ Upload OK!")
        print(f"🔗 https://huggingface.co/{repo_nome}")
    except Exception as e:
        print(f"\n❌ Erro: {e}")

# ============================================================
# MENU INTERATIVO
# ============================================================
def menu_interativo():
    global versao_atual

    while True:
        print(f"\n{'='*40}")
        print(f"✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print(f"{'='*40}")
        print(f"Escolha uma opção:")
        print(f"1️⃣  Treinar com novo dataset (continua do último checkpoint)")
        print(f"2️⃣  Testar a LLM treinada")
        print(f"3️⃣  Postar no Hugging Face")
        print(f"{'='*40}")

        escolha = input("\n👉 Opção (1, 2 ou 3): ").strip()

        if escolha == "1":
            novo_dataset = input("Nome do dataset no HF (ex: user/dataset): ").strip()
            if novo_dataset:
                ultimo = f"{DIR_BASE_MODELOS}_v{versao_atual}"
                executar_treinamento(
                    dataset_id=novo_dataset,
                    adaptadores_previos=ultimo if os.path.exists(ultimo) else None
                )
            else:
                print("❌ Dataset vazio.")

        elif escolha == "2":
            modo_teste()

        elif escolha == "3":
            upload_huggingface()

        else:
            print("❌ Opção inválida.")

# ============================================================
# MAIN
# ============================================================
print("""
╔══════════════════════════════════════════════════════════╗
║     🧠 PIPELINE DE FINE-TUNING COM UNSLOTH + LoRA       ║
║     Modelo: Skskskd/bliz-ia-PT-BR                      ║
║     Dataset: dominguesm/alpaca-data-pt-br               ║
║     Técnica: LoRA (r=16, α=32, dropout=0.05)           ║
║     Framework: Unsloth (sem 4-bit, otimizado)          ║
╚══════════════════════════════════════════════════════════╝
""")

sucesso = executar_treinamento(DATASET_INICIAL)

if sucesso:
    menu_interativo()
else:
    print("\n❌ Pipeline encerrado.")
