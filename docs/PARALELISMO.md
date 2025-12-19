# Decisão de Arquitetura: Task Única vs Multi-Task ECS

## 🧠 TL;DR (resposta curta)

👉 **SIM, uma única task parruda é a melhor opção agora.**

👉 Você já tem o framework certo, só falta paralelizar corretamente.

👉 ECS multi-task só vale a pena quando:
- `modelos ≫ 5`
- ou custos / quotas / isolamento forem críticos

---

## 🏗️ Comparação das Abordagens

### 🅰️ ECS com várias tasks (1 modelo por task)

**✅ Prós:**
- Isolamento forte
- Escala horizontal fácil
- Fault tolerance melhor

**❌ Contras (no seu cenário):**
- Muito overhead:
  - ECR
  - ECS Service
  - Networking
  - Orquestração
- Debug mais difícil
- Latência maior
- **Você não precisa disso agora**

### 🅱️ 1 TASK PARRUDA (multi-model, paralela)

**✅ Prós (PERFEITO pro seu caso):**
- Arquitetura simples
- Código reaproveitado
- Debug local ≈ produção
- Sem S3 intermediate por modelo
- Consenso imediato
- Mais barato
- Mais rápido para iterar

**❌ Contras:**
- Se a task morrer, tudo morre (aceitável agora)
- Limite de GPU/RAM (resolvido com instância grande)

---

## 🎯 O que você já tem (e está correto)

Você já tem:
- `LLMAnnotator`
- Cache
- Consenso
- Métricas
- Visualização

👉 **Ou seja: 90% pronto**

O que falta é paralelizar isso aqui:

```python
for model in self.models:
    for text in texts:
        annotate(text, model)
```

---

## 🧩 Arquitetura FINAL Recomendada (agora)

```
┌────────────────────────────┐
│ ECS Task (Fargate / EC2)   │
│                            │
│  ┌────────────────────┐    │
│  │ Async Orchestrator │    │
│  └─────────┬──────────┘    │
│            │                │
│  ┌─────────▼──────────┐    │
│  │ LLMAnnotator       │    │
│  │  • Model A         │    │
│  │  • Model B         │    │
│  │  • Model C         │    │
│  │  • Model D         │    │
│  │  • Model E         │    │
│  └─────────┬──────────┘    │
│            │                │
│  ┌─────────▼──────────┐    │
│  │ Consensus + Eval   │    │
│  └────────────────────┘    │
│                            │
│ → Salva tudo no S3         │
└────────────────────────────┘
```

---

## 🔥 COMO Paralelizar Corretamente

### Regra de Ouro

👉 **Paralelize por MODELO, não por TEXTO**

### Estratégia Técnica Recomendada

1️⃣ Async por modelo  
2️⃣ Semaphore para limitar concorrência  
3️⃣ Event loop único

### Exemplo Conceitual

```python
async def annotate_model(model, texts):
    sem = asyncio.Semaphore(10)

    async def annotate_one(text):
        async with sem:
            return await model.annotate(text)

    tasks = [annotate_one(t) for t in texts]
    return await asyncio.gather(*tasks)
```

Depois:

```python
results = await asyncio.gather(
    annotate_model(model_a, texts),
    annotate_model(model_b, texts),
    annotate_model(model_c, texts),
)
```

✔ Cada modelo com 10 requisições paralelas  
✔ Todos os modelos rodando juntos

---

## 🔧 Onde Isso Entra no Código Atual

**Arquivo alvo:**
```
LLMAnnotator.annotate_dataset()
```

Você vai:
- Criar uma versão async
- Manter fallback sync (bom pra debug)

---

## 💻 ECS Setup (Mínimo)

### Task Definition

```yaml
CPU: 16 vCPU
RAM: 64GB (ou mais)
Tipo:
  - EC2 se usar GPU
  - Fargate se for API-only
CMD: python run_experiment.py
```

### 💾 S3 (continua igual)

Você salva:
- CSV final
- Parquet
- JSON summary

Nada muda.

---

## 🧠 Recomendação FINAL

### Agora:

✅ 1 TASK PARRUDA  
✅ Paralelismo por modelo  
✅ Async + Semaphore  
✅ Consenso local

### Depois (se precisar escalar):

➡️ Migra para ECS multi-task sem reescrever o core

---

## 📝 Próximos Passos Práticos

1️⃣ Refatorar `LLMAnnotator.annotate_dataset()` para async paralelo  
2️⃣ Ajustar o `main()` para suportar async  
3️⃣ Definir limites ideais de concorrência por modelo