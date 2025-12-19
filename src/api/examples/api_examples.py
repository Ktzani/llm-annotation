"""
Exemplos de uso do cliente da API
"""

import requests
import time
import json
from typing import Dict, Any

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

API_BASE_URL = "http://localhost:8000"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def pretty_print(data: Dict[Any, Any]):
    """Imprime JSON de forma formatada"""
    print(json.dumps(data, indent=2, default=str))

def wait_for_experiment(experiment_id: str, check_interval: int = 5, timeout: int = 3600):
    """
    Aguarda a conclusão de um experimento
    
    Args:
        experiment_id: ID do experimento
        check_interval: Intervalo entre verificações (segundos)
        timeout: Timeout máximo (segundos)
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = requests.get(f"{API_BASE_URL}/experiments/{experiment_id}")
        
        if response.status_code != 200:
            print(f"Erro ao verificar status: {response.status_code}")
            return None
        
        status_data = response.json()
        status = status_data["status"]
        progress = status_data.get("progress", 0.0)
        message = status_data.get("message", "")
        
        print(f"Status: {status} | Progresso: {progress:.1%} | {message}")
        
        if status == "completed":
            print("\n✓ Experimento concluído!")
            return status_data
        elif status == "failed":
            print(f"\n✗ Experimento falhou: {message}")
            return status_data
        
        time.sleep(check_interval)
    
    print(f"\n⚠ Timeout após {timeout} segundos")
    return None

# =============================================================================
# EXEMPLO 1: EXPERIMENTO BÁSICO
# =============================================================================

def exemplo_basico():
    """Experimento básico com configuração mínima"""
    print("=" * 70)
    print("EXEMPLO 1: EXPERIMENTO BÁSICO")
    print("=" * 70)
    
    config = {
        "dataset_name": "sst2",
        "models": ["gemma3-4b"],
        "dataset_config": {
            "sample_size": 10,
        }
    }
    
    print("\nCriando experimento...")
    response = requests.post(f"{API_BASE_URL}/experiments", json=config)
    
    if response.status_code == 200:
        data = response.json()
        experiment_id = data["experiment_id"]
        print(f"Experimento criado: {experiment_id}")
        
        # Aguardar conclusão
        result = wait_for_experiment(experiment_id)
        
        if result and result["status"] == "completed":
            print("\nResultados:")
            pretty_print(result.get("results", {}))
    else:
        print(f"Erro: {response.status_code}")
        print(response.text)

# =============================================================================
# EXEMPLO 2: EXPERIMENTO AVANÇADO
# =============================================================================

def exemplo_avancado():
    """Experimento avançado com múltiplos modelos e repetições"""
    print("\n" + "=" * 70)
    print("EXEMPLO 2: EXPERIMENTO AVANÇADO")
    print("=" * 70)
    
    config = {
        "dataset_name": "sst2",
        "models": ["gemma3-4b", "mistral-7b"],

        "dataset_config": {
            "sample_size": 50,
            "random_state": 42
        },

        "annotation": {
            "num_repetitions_per_llm": 1,
            "model_strategy": "parallel",
            "rep_strategy": "parallel"
        },

        "cache": {
            "enabled": False
        },

        "results": {
            "save_intermediate": True
        }
    }
    
    print("\nConfiguração:")
    pretty_print(config)
    
    print("\nCriando experimento...")
    response = requests.post(f"{API_BASE_URL}/experiments", json=config)
    
    if response.status_code == 200:
        data = response.json()
        experiment_id = data["experiment_id"]
        print(f"Experimento criado: {experiment_id}")
        
        result = wait_for_experiment(experiment_id, check_interval=10)
        
        if result and result["status"] == "completed":
            print("\nResultados:")
            pretty_print(result.get("results", {}))
    else:
        print(f"Erro: {response.status_code}")
        print(response.text)

# =============================================================================
# EXEMPLO 3: PROMPT CUSTOMIZADO
# =============================================================================

def exemplo_prompt_customizado():
    """Experimento com prompt customizado"""
    print("\n" + "=" * 70)
    print("EXEMPLO 3: PROMPT CUSTOMIZADO")
    print("=" * 70)
    
    custom_prompt = """Analise o seguinte texto e classifique o sentimento.

Texto: {text}

Categorias possíveis: {categories}

Sua resposta deve ser APENAS uma das categorias acima, sem explicações adicionais.

Resposta:"""
    
    config = {
        "dataset_name": "sst2",
        "models": ["gemma3-4b"],
        "custom_prompt": custom_prompt,
        "dataset_config": {
            "sample_size": 20
        }
    }
    
    print("\nPrompt customizado:")
    print(custom_prompt)
    
    print("\nCriando experimento...")
    response = requests.post(f"{API_BASE_URL}/experiments", json=config)
    
    if response.status_code == 200:
        data = response.json()
        experiment_id = data["experiment_id"]
        print(f"Experimento criado: {experiment_id}")
        
        result = wait_for_experiment(experiment_id)
        
        if result and result["status"] == "completed":
            print("\nResultados:")
            pretty_print(result.get("results", {}))
    else:
        print(f"Erro: {response.status_code}")
        print(response.text)

# =============================================================================
# EXEMPLO 4: LISTAR E MONITORAR EXPERIMENTOS
# =============================================================================

def exemplo_listar_experimentos():
    """Lista todos os experimentos"""
    print("\n" + "=" * 70)
    print("EXEMPLO 4: LISTAR EXPERIMENTOS")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/experiments")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nTotal de experimentos: {data['total']}\n")
        
        for exp in data['experiments']:
            print(f"ID: {exp['experiment_id']}")
            print(f"Status: {exp['status']}")
            print(f"Criado em: {exp['created_at']}")
            print(f"Progresso: {exp.get('progress', 0):.1%}")
            print(f"Mensagem: {exp.get('message', 'N/A')}")
            print("-" * 70)
    else:
        print(f"Erro: {response.status_code}")
        print(response.text)

# =============================================================================
# EXEMPLO 5: DATASETS DISPONÍVEIS
# =============================================================================

def exemplo_listar_datasets():
    """Lista datasets disponíveis"""
    print("\n" + "=" * 70)
    print("EXEMPLO 5: DATASETS DISPONÍVEIS")
    print("=" * 70)
    
    response = requests.get(f"{API_BASE_URL}/datasets")
    
    if response.status_code == 200:
        data = response.json()
        print("\nDatasets disponíveis:")
        for dataset in data['datasets']:
            print(f"  - {dataset}")
    else:
        print(f"Erro: {response.status_code}")
        print(response.text)

# =============================================================================
# EXEMPLO 6: COMPARAÇÃO DE PROMPTS
# =============================================================================

def exemplo_comparacao_prompts():
    """Compara diferentes tipos de prompts"""
    print("\n" + "=" * 70)
    print("EXEMPLO 6: COMPARAÇÃO DE PROMPTS")
    print("=" * 70)
    
    prompt_types = ["base", "few_shot", "chain_of_thought", "simpler"]
    experiment_ids = []
    
    # Criar experimentos para cada tipo de prompt
    for prompt_type in prompt_types:
        config = {
            "dataset_name": "sst2",
            "models": ["gemma3-4b"],
            "prompt_type": prompt_type,
            "dataset_config": {
                "sample_size": 30,
                "random_state": 42
            },
            "results_dir": f"results/prompt_comparison/{prompt_type}"
        }
        
        print(f"\nCriando experimento com prompt: {prompt_type}")
        response = requests.post(f"{API_BASE_URL}/experiments", json=config)
        
        if response.status_code == 200:
            data = response.json()
            experiment_ids.append((prompt_type, data["experiment_id"]))
            print(f"  ✓ Experimento criado: {data['experiment_id']}")
        else:
            print(f"  ✗ Erro: {response.status_code}")
    
    # Aguardar todos os experimentos
    print("\n" + "=" * 70)
    print("AGUARDANDO CONCLUSÃO DOS EXPERIMENTOS...")
    print("=" * 70)
    
    results = {}
    for prompt_type, exp_id in experiment_ids:
        print(f"\nAguardando experimento {prompt_type}...")
        result = wait_for_experiment(exp_id, check_interval=5)
        
        if result and result["status"] == "completed":
            results[prompt_type] = result.get("results", {})
    
    # Comparar resultados
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE RESULTADOS")
    print("=" * 70)
    
    for prompt_type, result in results.items():
        print(f"\n{prompt_type.upper()}:")
        if "metrics" in result:
            metrics = result["metrics"][0] if result["metrics"] else {}
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"  F1 Score: {metrics.get('f1_score', 'N/A')}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Executa todos os exemplos"""
    print("\n" + "=" * 70)
    print("API CLIENT - EXEMPLOS DE USO")
    print("=" * 70)
    
    # Health check
    print("\nVerificando saúde da API...")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        print("✓ API está online")
    else:
        print("✗ API não está respondendo")
        return
    
    # Menu
    while True:
        print("\n" + "=" * 70)
        print("ESCOLHA UM EXEMPLO:")
        print("=" * 70)
        print("1. Experimento Básico")
        print("2. Experimento Avançado")
        print("3. Prompt Customizado")
        print("4. Listar Experimentos")
        print("5. Listar Datasets")
        print("6. Comparação de Prompts")
        print("0. Sair")
        print("=" * 70)
        
        choice = input("\nEscolha: ")
        
        if choice == "1":
            exemplo_basico()
        elif choice == "2":
            exemplo_avancado()
        elif choice == "3":
            exemplo_prompt_customizado()
        elif choice == "4":
            exemplo_listar_experimentos()
        elif choice == "5":
            exemplo_listar_datasets()
        elif choice == "6":
            exemplo_comparacao_prompts()
        elif choice == "0":
            print("\nAté logo!")
            break
        else:
            print("\nOpção inválida!")
        
        input("\nPressione ENTER para continuar...")

if __name__ == "__main__":
    main()