# ============================================================
# CÉLULA 1: INSTALAÇÃO - Execute esta célula primeiro
# ============================================================
!pip install --upgrade pip

# Instalar Unsloth
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Instalar dependências sem conflitos
!pip install --no-deps trl peft accelerate xformers

# Instalar outras dependências
!pip install -U transformers datasets scipy huggingface_hub

print("\n" + "="*60)
print("✅ INSTALAÇÃO CONCLUÍDA!")
print("="*60)
print("⚠️  AGORA VOCÊ PRECISA REINICIAR O RUNTIME:")
print("   Menu → Ambiente de execução → Reiniciar sessão")
print("   OU pressione: Ctrl + M + .")
print("="*60)
