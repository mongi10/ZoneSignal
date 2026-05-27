"""
ZoneSignal - Market History Engine
Téléchargement intelligent avec cache permanent et mises à jour incrémentales
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.core.cache import cache_manager
from app.models.market import MarketHistory


class MarketHistoryEngine:
    """
    Moteur de gestion des historiques de marché
    
    Règles critiques:
    - Première exécution: téléchargement complet + cache permanent
    - Exécutions suivantes: mises à jour incrémentales uniquement
    - Mutualisation avec ZoneFlow (vérification cache existant)
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.cache = cache_manager
    
    # ═══════════════════════════════════════════════════════════════
    # TÉLÉCHARGEMENT INTELLIGENT
    # ═══════════════════════════════════════════════════════════════
    
    def sync_symbol(
        self, 
        symbol: str, 
        force_full: bool = False
    ) -> Dict[str, Any]:
        """
        Synchronise les données d'un symbole
        
        Args:
            symbol: Ticker du symbole (ex: "^GSPC")
            force_full: Force un re-téléchargement complet
            
        Returns:
            Statistiques de synchronisation
        """
        result = {
            "symbol": symbol,
            "action": None,
            "rows_added": 0,
            "rows_updated": 0,
            "errors": []
        }
        
        try:
            # Vérifier le cache ZoneFlow existant
            cached_last_date = self.cache.get_last_update("market_history", symbol)
            db_last_date = self._get_last_db_date(symbol)
            
            if force_full or (not cached_last_date and not db_last_date):
                # Premier téléchargement complet
                result["action"] = "full_download"
                stats = self._full_download(symbol)
                result.update(stats)
            else:
                # Mise à jour incrémentale
                result["action"] = "incremental_update"
                last_date = db_last_date or datetime.strptime(cached_last_date, "%Y-%m-%d").date()
                stats = self._incremental_update(symbol, last_date)
                result.update(stats)
            
            # Mettre à jour le timestamp du cache
            self.cache.set_last_update(
                "market_history", 
                symbol, 
                date.today().isoformat()
            )
            
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def _full_download(self, symbol: str) -> Dict[str, int]:
        """Téléchargement complet de l'historique (plusieurs décennies)"""
        
        # Télécharger depuis Yahoo Finance (max historique disponible)
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="max", auto_adjust=True)
        
        if df.empty:
            raise ValueError(f"Aucune donnée disponible pour {symbol}")
        
        # Calculer les métriques dérivées
        df = self._compute_derived_metrics(df)
        
        # Insérer en base
        rows_added = self._insert_history_data(symbol, df)
        
        # Mettre en cache
        self.cache.set_permanent(
            "market_history",
            f"{symbol}_full",
            df.to_dict()
        )
        
        return {"rows_added": rows_added, "rows_updated": 0}
    
    def _incremental_update(
        self, 
        symbol: str, 
        last_date: date
    ) -> Dict[str, int]:
        """Mise à jour incrémentale depuis la dernière date"""
        
        # Télécharger uniquement les nouvelles données
        start_date = last_date + timedelta(days=1)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, auto_adjust=True)
        
        if df.empty:
            return {"rows_added": 0, "rows_updated": 0}
        
        # Calculer les métriques dérivées
        df = self._compute_derived_metrics(df)
        
        # Insérer les nouvelles données
        rows_added = self._insert_history_data(symbol, df)
        
        # Mettre à jour le cache (append)
        self._update_cache_incremental(symbol, df)
        
        return {"rows_added": rows_added, "rows_updated": 0}
    
    # ═══════════════════════════════════════════════════════════════
    # CALCUL DES MÉTRIQUES DÉRIVÉES
    # ═══════════════════════════════════════════════════════════════
    
    def _compute_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule toutes les métriques dérivées"""
        
        df = df.copy()
        
        # Returns
        df['daily_return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatilité
        df['volatility_20d'] = df['daily_return'].rolling(20).std() * np.sqrt(252)
        df['volatility_60d'] = df['daily_return'].rolling(60).std() * np.sqrt(252)
        
        # ATR
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(14).mean()
        
        # Drawdown
        rolling_max = df['Close'].expanding().max()
        df['drawdown_from_high'] = (df['Close'] - rolling_max) / rolling_max
        
        # Days since high
        df['is_new_high'] = df['Close'] >= rolling_max
        df['days_since_high'] = (~df['is_new_high']).groupby(
            df['is_new_high'].cumsum()
        ).cumcount()
        
        # Dollar volume
        df['dollar_volume'] = df['Close'] * df['Volume']
        
        # Relative volume (vs 20-day average)
        df['relative_volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Percent above MAs (pour index, approximation)
        df['ma50'] = df['Close'].rolling(50).mean()
        df['ma200'] = df['Close'].rolling(200).mean()
        df['percent_above_ma50'] = (df['Close'] > df['ma50']).astype(float)
        df['percent_above_ma200'] = (df['Close'] > df['ma200']).astype(float)
        
        return df
    
    # ═══════════════════════════════════════════════════════════════
    # OPÉRATIONS BASE DE DONNÉES
    # ═══════════════════════════════════════════════════════════════
    
    def _get_last_db_date(self, symbol: str) -> Optional[date]:
        """Récupère la dernière date en base pour un symbole"""
        result = self.db.query(func.max(MarketHistory.date)).filter(
            MarketHistory.symbol == symbol
        ).scalar()
        return result
    
    def _insert_history_data(
        self, 
        symbol: str, 
        df: pd.DataFrame
    ) -> int:
        """Insère les données historiques en base"""
        
        rows_added = 0
        
        for idx, row in df.iterrows():
            # Vérifier si existe déjà
            existing = self.db.query(MarketHistory).filter(
                MarketHistory.symbol == symbol,
                MarketHistory.date == idx.date()
            ).first()
            
            if existing:
                continue
            
            record = MarketHistory(
                symbol=symbol,
                date=idx.date(),
                open=row.get('Open'),
                high=row.get('High'),
                low=row.get('Low'),
                close=row.get('Close'),
                adj_close=row.get('Close'),  # auto_adjust=True
                volume=row.get('Volume'),
                daily_return=row.get('daily_return'),
                log_return=row.get('log_return'),
                volatility_20d=row.get('volatility_20d'),
                volatility_60d=row.get('volatility_60d'),
                atr_14=row.get('atr_14'),
                drawdown_from_high=row.get('drawdown_from_high'),
                days_since_high=row.get('days_since_high'),
                dollar_volume=row.get('dollar_volume'),
                relative_volume=row.get('relative_volume'),
                percent_above_ma50=row.get('percent_above_ma50'),
                percent_above_ma200=row.get('percent_above_ma200')
            )
            
            self.db.add(record)
            rows_added += 1
        
        self.db.commit()
        return rows_added
    
    def _update_cache_incremental(self, symbol: str, df: pd.DataFrame):
        """Met à jour le cache de manière incrémentale"""
        # Récupérer le cache existant
        cached = self.cache.get_permanent("market_history", f"{symbol}_full")
        
        if cached:
            existing_df = pd.DataFrame(cached)
            # Append les nouvelles données
            combined = pd.concat([existing_df, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            self.cache.set_permanent(
                "market_history",
                f"{symbol}_full",
                combined.to_dict()
            )
    
    # ═══════════════════════════════════════════════════════════════
    # RÉCUPÉRATION DES DONNÉES
    # ═══════════════════════════════════════════════════════════════
    
    def get_history(
        self,
        symbol: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Récupère l'historique d'un symbole
        
        Priorité: Cache > Base de données
        """
        
        if use_cache:
            cached = self.cache.get_permanent("market_history", f"{symbol}_full")
            if cached:
                df = pd.DataFrame(cached)
                df.index = pd.to_datetime(df.index)
                
                if start_date:
                    df = df[df.index >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]
                
                return df
        
        # Fallback sur la base de données
        query = self.db.query(MarketHistory).filter(
            MarketHistory.symbol == symbol
        )
        
        if start_date:
            query = query.filter(MarketHistory.date >= start_date)
        if end_date:
            query = query.filter(MarketHistory.date <= end_date)
        
        query = query.order_by(MarketHistory.date)
        records = query.all()
        
        if not records:
            return pd.DataFrame()
        
        data = [{
            'date': r.date,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume,
            'daily_return': r.daily_return,
            'volatility_20d': r.volatility_20d,
            'volatility_60d': r.volatility_60d,
            'drawdown_from_high': r.drawdown_from_high
        } for r in records]
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df
    
    def sync_all_indices(self) -> List[Dict[str, Any]]:
        """Synchronise tous les indices configurés"""
        results = []
        
        for symbol in settings.SUPPORTED_INDICES:
            result = self.sync_symbol(symbol)
            results.append(result)
        
        return results
