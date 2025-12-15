"""
Response Processor - Processa e extrai respostas das LLMs
"""

from typing import List
from loguru import logger
import re

class ResponseProcessor:
    """
    Processa respostas das LLMs
    Responsabilidades: extrair categorias, validar respostas, normalizar output
    """
    
    def __init__(self, categories: List[int]):
        """
        Args:
            categories: Lista de categorias válidas
        """
        self.categories = categories
        logger.debug(f"ResponseProcessor inicializado com {len(categories)} categorias")
    
    def extract_category(self, response: str) -> int:
        """
        Extrai categoria da resposta da LLM
        
        Args:
            response: Resposta completa da LLM
            
        Returns:
            Categoria extraída (int) ou -1
        """
        if response is None or response.strip() == "":
            logger.warning("Resposta é None ou ''")
            return -1
        
        response = response.strip()
        
        # Extrair "CLASSIFICATION:" se presente
        if "CLASSIFICATION:" in response:
            response = response.split("CLASSIFICATION:")[-1].strip()
        
        # Primeiro tenta converter direto
        try:
            value = int(response)
        except Exception:
            # Caso falhe, tenta extrair o primeiro número usando regex
            match = re.search(r"-?\d+", response)
            if match:
                try:
                    value = int(match.group())
                except Exception:
                    logger.warning(f"Regex encontrou um número, mas falhou ao converter: {match.group()}")
                    return -1
            else:
                logger.warning(f"Falha ao converter resposta para inteiro e nenhum número encontrado: '{response[:50]}'")
                return -1
        
        # Verificar se está nas categorias válidas
        if value in self.categories:
            return value
        
        logger.warning(f"Categoria {value} não está na lista de válidas: {self.categories}")
        return -1
    
    def validate_response(self, response: str) -> bool:
        """
        Valida se a resposta é válida
        
        Args:
            response: Resposta da LLM
            
        Returns:
            True se válida
        """
        category = self.extract_category(response)
        return category in self.categories
