"""
Cache Manager - Gerencia cache de respostas
"""

import json
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

from src.config.datasets_config import CACHE_DIR


class CacheManager:
    """
    Gerencia cache de respostas das LLMs
    Responsabilidades: salvar, carregar, verificar cache
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR):
        """
        Args:
            cache_dir: Diretório para cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_file = self.cache_dir / "response_cache.json"
        self.cache = self._load()
        logger.debug(f"Cache inicializado: {self.cache_file}")
    
    def _load(self) -> Dict[str, Any]:
        """Carrega cache do disco"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"Cache carregado: {len(cache)} entradas")
                return cache
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}")
                return {}
        return {}
    
    def save(self):
        """Salva cache no disco"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cache salvo: {len(self.cache)} entradas")
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
    
    def get_key(self, model: str, text: str, params: Dict) -> str:
        """
        Gera chave única para cache
        
        Args:
            model: Nome do modelo
            text: Texto
            params: Parâmetros
            
        Returns:
            Chave MD5
        """
        content = f"{model}|{text}|{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[str]:
        """
        Busca resposta no cache
        
        Args:
            key: Chave do cache
            
        Returns:
            Resposta em cache ou None
        """
        return self.cache.get(key)
    
    def set(self, key: str, value: str):
        """
        Adiciona resposta ao cache
        
        Args:
            key: Chave do cache
            value: Resposta
        """
        self.cache[key] = value
        
        # Auto-save a cada 10 entradas
        if len(self.cache) % 10 == 0:
            self.save()
    
    def clear(self):
        """Limpa o cache"""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache limpo")
    
    def stats(self) -> Dict[str, int]:
        """
        Retorna estatísticas do cache
        
        Returns:
            Dicionário com estatísticas
        """
        return {
            "total_entries": len(self.cache),
            "cache_size_mb": self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
        }


class LangChainCacheManager:
    """
    Gerencia cache nativo do LangChain (SQLite)
    """
    
    def __init__(self, cache_dir: str = "./cache", enabled: bool = True):
        """
        Args:
            cache_dir: Diretório para cache
            enabled: Se True, ativa cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.enabled = enabled
        
        if enabled:
            self._setup()
    
    def _setup(self):
        """Configura cache SQLite do LangChain"""
        try:
            from langchain.cache import SQLiteCache
            from langchain.globals import set_llm_cache
            
            cache_file = self.cache_dir / "langchain_cache.db"
            set_llm_cache(SQLiteCache(database_path=str(cache_file)))
            logger.info(f"Cache LangChain ativado: {cache_file}")
        except ImportError:
            logger.warning("langchain.cache não disponível - cache desativado")
