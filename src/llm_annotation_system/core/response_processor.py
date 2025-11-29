"""
Response Processor - Processa e extrai respostas das LLMs
"""

from typing import List
from loguru import logger


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
    
    def extract_category(self, response: str) -> int | str:
        """
        Extrai categoria da resposta da LLM
        
        Args:
            response: Resposta completa da LLM
            
        Returns:
            Categoria extraída (int) ou "ERROR"
        """
        if response is None:
            logger.warning("Resposta é None")
            return "ERROR"
        
        response = response.strip()
        
        if not self.validate_response(response):
            logger.warning(f"Resposta inválida - Não possui a categoria necessária: '{response[:50]}'")
            return "ERROR"
        
        # Extrair "CLASSIFICATION:" se presente
        if "CLASSIFICATION:" in response:
            response = response.split("CLASSIFICATION:")[-1].strip()
        
        # Tentar converter para inteiro
        try:
            value = int(response)
        except Exception:
            logger.warning(f"Falha ao converter resposta para inteiro: '{response[:50]}'")
            return "ERROR"
        
        # Verificar se está nas categorias válidas
        if value in self.categories:
            return value
        
        logger.warning(f"Categoria {value} não está na lista de válidas: {self.categories}")
        return "ERROR"
    
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
