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
    
    def __init__(self, categories: List[str]):
        """
        Args:
            categories: Lista de categorias válidas
        """
        self.categories = categories
        self.categories_lower = [c.lower() for c in categories]
        logger.debug(f"ResponseProcessor inicializado com {len(categories)} categorias")
    
    def extract_category(self, response: str) -> str:
        """
        Extrai categoria da resposta da LLM
        
        Args:
            response: Resposta completa da LLM
            
        Returns:
            Categoria extraída
        """
        if response is None:
            logger.warning("Resposta é None")
            return "ERROR"
        
        response = response.strip()
        
        # Extrair "CLASSIFICATION:" se presente
        if "CLASSIFICATION:" in response:
            response = response.split("CLASSIFICATION:")[-1].strip()
        
        # Normalizar
        response_clean = response.lower().strip('.,!?;:"\'')
        
        # Correspondência exata
        for i, category_lower in enumerate(self.categories_lower):
            if category_lower == response_clean:
                return self.categories[i]
        
        # Correspondência parcial
        for i, category_lower in enumerate(self.categories_lower):
            if category_lower in response_clean or response_clean in category_lower:
                return self.categories[i]
        
        # Não encontrou
        logger.warning(f"Categoria não encontrada em: '{response[:50]}'")
        return response[:50]  # Limitar tamanho
    
    def validate_response(self, response: str) -> bool:
        """
        Valida se a resposta é válida
        
        Args:
            response: Resposta da LLM
            
        Returns:
            True se válida
        """
        if response is None or response == "ERROR":
            return False
        
        category = self.extract_category(response)
        return category in self.categories
