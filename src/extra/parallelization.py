from multiprocessing import Pool, Manager, cpu_count
from collections import Counter

@staticmethod
def process_model(args):
    """Processa todos os textos para um único modelo"""
    annotator, model, texts, num_repetitions, use_cache, results_queue = args
    
    for idx, text in enumerate(texts):
        annotations = annotator.annotate_single(
            text=text,
            model=model,
            num_repetitions=num_repetitions,
            use_cache=use_cache
        )
        
        # Envia resultado para a fila principal
        results_queue.put((idx, model, annotations))
    
    return model  # Retorna apenas confirmação de conclusão


def annotate_dataset(
    self,
    texts: List[str],
    num_repetitions: Optional[int] = None,
    save_intermediate: bool = True,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Anota dataset completo com paralelização por modelo
    """
    if num_repetitions is None:
        num_repetitions = EXPERIMENT_CONFIG.get("num_repetitions_per_llm", 3)
    
    total_annotations = len(texts) * len(self.models) * num_repetitions
    
    logger.info(f"Iniciando anotação PARALELA")
    logger.info(f"Textos: {len(texts)} | Modelos: {len(self.models)} | Repetições: {num_repetitions}")
    logger.info(f"Total de anotações: {total_annotations}")
    
    n_workers = min(len(self.models), cpu_count())
    logger.info(f"Usando {n_workers} processos (1 por modelo)")
    
    # Estrutura para armazenar resultados: {text_idx: {model: annotations}}
    text_annotations = {idx: {} for idx in range(len(texts))}
    completed_texts = set()
    
    with Manager() as manager:
        results_queue = manager.Queue()
        
        # Argumentos para cada worker
        args_list = [
            (self, model, texts, num_repetitions, use_cache, results_queue)
            for model in self.models
        ]
        
        total_tasks = len(texts) * len(self.models)
        
        with Pool(processes=n_workers) as pool:
            # Inicia processamento assíncrono
            async_result = pool.map_async(self.process_model, args_list)
            
            # Coleta resultados conforme chegam
            with tqdm(total=total_tasks, desc="Processando textos") as pbar:
                completed_tasks = 0
                
                while completed_tasks < total_tasks:
                    try:
                        # Pega resultado da fila
                        idx, model, annotations = results_queue.get(timeout=0.1)
                        
                        # Armazena as anotações
                        text_annotations[idx][model] = annotations
                        completed_tasks += 1
                        pbar.update(1)
                        
                        # Verifica se o texto está completo (todos os modelos processaram)
                        if len(text_annotations[idx]) == len(self.models):
                            completed_texts.add(idx)
                            
                            # Salvar intermediário a cada 10 textos COMPLETOS
                            if save_intermediate and len(completed_texts) % 10 == 0:
                                self._save_intermediate_results(
                                    texts, 
                                    text_annotations, 
                                    completed_texts,
                                    len(completed_texts)
                                )
                    
                    except:
                        if async_result.ready():
                            break
                        continue
            
            # Garante que tudo foi processado
            async_result.get()
    
    # Montar DataFrame final
    results = []
    
    for idx in range(len(texts)):
        text_results = {
            'text_id': idx,
            'text': texts[idx][:200],
        }
        
        for model in self.models:
            annotations = text_annotations[idx][model]
            
            # Salvar repetições
            for rep_idx, annotation in enumerate(annotations):
                text_results[f"{model}_rep{rep_idx+1}"] = annotation
            
            # Consenso interno
            most_common = Counter(annotations).most_common(1)[0]
            text_results[f"{model}_consensus"] = most_common[0]
            text_results[f"{model}_consensus_score"] = most_common[1] / len(annotations)
        
        results.append(text_results)
    
    # Salvar cache
    self.cache_manager.save()
    
    # DataFrame final
    df = pd.DataFrame(results)
    
    output_file = self.results_dir / "annotations_complete.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    logger.success(f"Anotações completas salvas: {output_file}")
    
    return df


def _save_intermediate_results(
    self,
    texts: List[str],
    text_annotations: dict,
    completed_texts: set,
    num_completed: int
):
    """Salva resultados intermediários apenas dos textos completos"""
    results = []
    
    # Ordena os índices completos
    for idx in sorted(completed_texts):
        text_results = {
            'text_id': idx,
            'text': texts[idx][:200],
        }
        
        for model in self.models:
            annotations = text_annotations[idx][model]
            
            # Salvar repetições
            for rep_idx, annotation in enumerate(annotations):
                text_results[f"{model}_rep{rep_idx+1}"] = annotation
            
            # Consenso interno
            most_common = Counter(annotations).most_common(1)[0]
            text_results[f"{model}_consensus"] = most_common[0]
            text_results[f"{model}_consensus_score"] = most_common[1] / len(annotations)
        
        results.append(text_results)
    
    df_temp = pd.DataFrame(results)
    df_temp.to_csv(
        self.results_dir / f"intermediate_{num_completed}.csv",
        index=False,
        encoding='utf-8'
    )
    logger.debug(f"Salvos {num_completed} textos completos (intermediário)")