"""
ZoneSignal - Système de cache Redis
Mutualisation avec ZoneFlow, cache permanent après premier téléchargement
"""

import redis
import json
import pickle
from typing import Any, Optional
from datetime import timedelta
from app.core.config import settings


class CacheManager:
    """
    Gestionnaire de cache institutionnel
    - Cache permanent pour historiques
    - Cache temporaire pour prévisions
    - Invalidation intelligente
    """
    
    def __init__(self):
        self.redis_client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=False
        )
        
        # Préfixes de cache
        self.PREFIXES = {
            "market_history": "mh:",      # Cache permanent
            "macro_data": "macro:",        # Cache semi-permanent (24h)
            "forecast": "fc:",             # Cache court (1h)
            "regime": "reg:",              # Cache court (1h)
            "seasonality": "seas:",        # Cache permanent
            "model_weights": "mw:",        # Cache permanent
            "backtest": "bt:"              # Cache permanent
        }
    
    def _get_key(self, prefix: str, key: str) -> str:
        return f"zonesignal:{prefix}{key}"
    
    # ═══════════════════════════════════════════════════════════════
    # CACHE PERMANENT (Historiques, Saisonnalité, Modèles)
    # ═══════════════════════════════════════════════════════════════
    
    def set_permanent(self, category: str, key: str, data: Any) -> bool:
        """Cache permanent - pas d'expiration"""
        try:
            full_key = self._get_key(self.PREFIXES.get(category, ""), key)
            serialized = pickle.dumps(data)
            self.redis_client.set(full_key, serialized)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def get_permanent(self, category: str, key: str) -> Optional[Any]:
        """Récupération cache permanent"""
        try:
            full_key = self._get_key(self.PREFIXES.get(category, ""), key)
            data = self.redis_client.get(full_key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════
    # CACHE TEMPORAIRE (Prévisions, Régimes)
    # ═══════════════════════════════════════════════════════════════
    
    def set_with_ttl(
        self, 
        category: str, 
        key: str, 
        data: Any, 
        ttl_seconds: int = 3600
    ) -> bool:
        """Cache avec expiration"""
        try:
            full_key = self._get_key(self.PREFIXES.get(category, ""), key)
            serialized = pickle.dumps(data)
            self.redis_client.setex(full_key, ttl_seconds, serialized)
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    def get_with_ttl(self, category: str, key: str) -> Optional[Any]:
        """Récupération cache temporaire"""
        return self.get_permanent(category, key)
    
    # ═══════════════════════════════════════════════════════════════
    # VÉRIFICATION & INVALIDATION
    # ═══════════════════════════════════════════════════════════════
    
    def exists(self, category: str, key: str) -> bool:
        """Vérifie si une clé existe"""
        full_key = self._get_key(self.PREFIXES.get(category, ""), key)
        return self.redis_client.exists(full_key) > 0
    
    def get_last_update(self, category: str, key: str) -> Optional[str]:
        """Récupère la date de dernière mise à jour"""
        meta_key = f"{category}_meta:{key}:last_update"
        data = self.redis_client.get(meta_key)
        return data.decode() if data else None
    
    def set_last_update(self, category: str, key: str, date: str) -> bool:
        """Enregistre la date de dernière mise à jour"""
        meta_key = f"{category}_meta:{key}:last_update"
        self.redis_client.set(meta_key, date)
        return True
    
    def invalidate(self, category: str, key: str) -> bool:
        """Invalide une entrée de cache"""
        full_key = self._get_key(self.PREFIXES.get(category, ""), key)
        self.redis_client.delete(full_key)
        return True
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalide toutes les clés correspondant à un pattern"""
        keys = self.redis_client.keys(f"zonesignal:{pattern}*")
        if keys:
            return self.redis_client.delete(*keys)
        return 0


# Instance singleton
cache_manager = CacheManager()
