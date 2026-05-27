"""
ZoneSignal - Seasonality Engine
Analyse avancée des patterns saisonniers et cycliques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from datetime import date, datetime
import calendar


@dataclass
class SeasonalityPattern:
    """Pattern de saisonnalité détecté"""
    pattern_type: str
    period_key: str
    avg_return: float
    median_return: float
    win_rate: float
    avg_volatility: float
    sample_size: int
    t_statistic: float
    p_value: float
    is_significant: bool
    historical_data: List[float]


@dataclass
class SeasonalityForecast:
    """Prévision basée sur la saisonnalité"""
    expected_return: float
    confidence: float
    primary_patterns: List[SeasonalityPattern]
    composite_score: float
    historical_analog_returns: List[float]


class SeasonalityEngine:
    """
    Moteur d'analyse de saisonnalité institutionnel
    
    Patterns analysés:
    - Monthly seasonality
    - Quarterly seasonality  
    - Presidential cycle (4 ans)
    - Election years
    - Santa rally
    - Sell in May
    - January effect
    - Options expiration
    - Earnings seasons
    - Tax-loss selling
    - Turn of month
    - Holiday effects
    """
    
    def __init__(self):
        self.significance_level = 0.10  # Seuil de significativité
        self.min_samples = 10  # Minimum d'échantillons
        
        # Définition des patterns connus
        self.known_patterns = {
            "january_effect": "Tendance haussière en janvier",
            "santa_rally": "Rally des derniers jours de décembre",
            "sell_in_may": "Faiblesse mai-octobre vs nov-avril",
            "september_effect": "Historiquement le pire mois",
            "turn_of_month": "Derniers et premiers jours du mois",
            "options_expiration": "Triple/quadruple witching",
            "fomc_drift": "Drift avant les réunions FOMC",
            "earnings_season": "Périodes de résultats trimestriels"
        }
    
    # ═══════════════════════════════════════════════════════════════
    # ANALYSE MENSUELLE
    # ═══════════════════════════════════════════════════════════════
    
    def analyze_monthly_seasonality(
        self,
        df: pd.DataFrame,
        min_years: int = 10
    ) -> Dict[int, SeasonalityPattern]:
        """
        Analyse la saisonnalité mensuelle
        
        Returns:
            Dict avec les statistiques pour chaque mois (1-12)
        """
        
        df = df.copy()
        df['date'] = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Calculer les returns mensuels
        monthly_returns = df.groupby(['year', 'month'])['daily_return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        monthly_returns.columns = ['year', 'month', 'return']
        
        patterns = {}
        
        for month in range(1, 13):
            month_data = monthly_returns[monthly_returns['month'] == month]['return'].values
            
            if len(month_data) < self.min_samples:
                continue
            
            # Statistiques
            avg_return = np.mean(month_data)
            median_return = np.median(month_data)
            win_rate = np.mean(month_data > 0)
            avg_vol = np.std(month_data)
            
            # Test de significativité (t-test vs 0)
            t_stat, p_value = stats.ttest_1samp(month_data, 0)
            
            patterns[month] = SeasonalityPattern(
                pattern_type="monthly",
                period_key=calendar.month_name[month],
                avg_return=float(avg_return),
                median_return=float(median_return),
                win_rate=float(win_rate),
                avg_volatility=float(avg_vol),
                sample_size=len(month_data),
                t_statistic=float(t_stat),
                p_value=float(p_value),
                is_significant=p_value < self.significance_level,
                historical_data=month_data.tolist()
            )
        
        return patterns
    
    # ═══════════════════════════════════════════════════════════════
    # CYCLE PRÉSIDENTIEL (4 ANS)
    # ═══════════════════════════════════════════════════════════════
    
    def analyze_presidential_cycle(
        self,
        df: pd.DataFrame
    ) -> Dict[int, SeasonalityPattern]:
        """
        Analyse le cycle présidentiel US (année 1-4 du mandat)
        
        Historiquement:
        - Année 1 (post-élection): volatilité, réformes
        - Année 2 (midterm): souvent faible
        - Année 3 (pré-élection): historiquement fort
        - Année 4 (élection): incertain puis rally
        """
        
        df = df.copy()
        df['date'] = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'])
        df['year'] = df['date'].dt.year
        
        # Années d'élection US connues (cycle de 4 ans)
        # 2020, 2016, 2012, 2008, etc.
        df['cycle_year'] = ((df['year'] - 2020) % 4) + 1
        # 1 = post-élection, 2 = midterm, 3 = pré-élection, 4 = élection
        
        # Returns annuels par année du cycle
        yearly_returns = df.groupby(['year', 'cycle_year'])['daily_return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        yearly_returns.columns = ['year', 'cycle_year', 'return']
        
        cycle_names = {
            1: "Post-Election (Year 1)",
            2: "Midterm (Year 2)", 
            3: "Pre-Election (Year 3)",
            4: "Election (Year 4)"
        }
        
        patterns = {}
        
        for cycle_year in range(1, 5):
            cycle_data = yearly_returns[yearly_returns['cycle_year'] == cycle_year]['return'].values
            
            if len(cycle_data) < 5:
                continue
            
            avg_return = np.mean(cycle_data)
            t_stat, p_value = stats.ttest_1samp(cycle_data, 0)
            
            patterns[cycle_year] = SeasonalityPattern(
                pattern_type="presidential_cycle",
                period_key=cycle_names[cycle_year],
                avg_return=float(avg_return),
                median_return=float(np.median(cycle_data)),
                win_rate=float(np.mean(cycle_data > 0)),
                avg_volatility=float(np.std(cycle_data)),
                sample_size=len(cycle_data),
                t_statistic=float(t_stat),
                p_value=float(p_value),
                is_significant=p_value < self.significance_level,
                historical_data=cycle_data.tolist()
            )
        
        return patterns
    
    # ═══════════════════════════════════════════════════════════════
    # PATTERNS SPÉCIAUX
    # ═══════════════════════════════════════════════════════════════
    
    def analyze_santa_rally(
        self,
        df: pd.DataFrame
    ) -> SeasonalityPattern:
        """
        Analyse le 'Santa Claus Rally'
        
        Période: derniers 5 jours de décembre + 2 premiers de janvier
        """
        
        df = df.copy()
        df['date'] = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        
        # Identifier les périodes de Santa Rally
        santa_periods = []
        
        for year in df['year'].unique():
            # Derniers 5 jours de décembre
            dec_mask = (df['year'] == year) & (df['month'] == 12) & (df['day'] >= 26)
            # Premiers 2 jours de janvier suivant
            jan_mask = (df['year'] == year + 1) & (df['month'] == 1) & (df['day'] <= 2)
            
            period_returns = df[dec_mask | jan_mask]['daily_return'].values
            if len(period_returns) > 0:
                period_return = (1 + period_returns).prod() - 1
                santa_periods.append(period_return)
        
        santa_returns = np.array(santa_periods)
        
        if len(santa_returns) < self.min_samples:
            return None
        
        t_stat, p_value = stats.ttest_1samp(santa_returns, 0)
        
        return SeasonalityPattern(
            pattern_type="santa_rally",
            period_key="Dec 26 - Jan 2",
            avg_return=float(np.mean(santa_returns)),
            median_return=float(np.median(santa_returns)),
            win_rate=float(np.mean(santa_returns > 0)),
            avg_volatility=float(np.std(santa_returns)),
            sample_size=len(santa_returns),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=p_value < self.significance_level,
            historical_data=santa_returns.tolist()
        )
    
    def analyze_sell_in_may(
        self,
        df: pd.DataFrame
    ) -> Dict[str, SeasonalityPattern]:
        """
        Analyse 'Sell in May and Go Away'
        
        Compare Nov-Avril vs Mai-Octobre
        """
        
        df = df.copy()
        df['date'] = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Période "favorable": Nov-Avril
        df['is_favorable'] = df['month'].isin([11, 12, 1, 2, 3, 4])
        
        # Returns semestriels
        favorable_returns = []
        unfavorable_returns = []
        
        for year in df['year'].unique():
            # Favorable: Nov year-1 to April year
            fav_mask = (
                ((df['year'] == year - 1) & (df['month'].isin([11, 12]))) |
                ((df['year'] == year) & (df['month'].isin([1, 2, 3, 4])))
            )
            fav_ret = df[fav_mask]['daily_return'].values
            if len(fav_ret) > 0:
                favorable_returns.append((1 + fav_ret).prod() - 1)
            
            # Unfavorable: May-Oct year
            unfav_mask = (df['year'] == year) & (df['month'].isin([5, 6, 7, 8, 9, 10]))
            unfav_ret = df[unfav_mask]['daily_return'].values
            if len(unfav_ret) > 0:
                unfavorable_returns.append((1 + unfav_ret).prod() - 1)
        
        patterns = {}
        
        fav_arr = np.array(favorable_returns)
        if len(fav_arr) >= self.min_samples:
            t_stat, p_value = stats.ttest_1samp(fav_arr, 0)
            patterns["favorable"] = SeasonalityPattern(
                pattern_type="sell_in_may",
                period_key="Nov-April (Favorable)",
                avg_return=float(np.mean(fav_arr)),
                median_return=float(np.median(fav_arr)),
                win_rate=float(np.mean(fav_arr > 0)),
                avg_volatility=float(np.std(fav_arr)),
                sample_size=len(fav_arr),
                t_statistic=float(t_stat),
                p_value=float(p_value),
                is_significant=p_value < self.significance_level,
                historical_data=fav_arr.tolist()
            )
        
        unfav_arr = np.array(unfavorable_returns)
        if len(unfav_arr) >= self.min_samples:
            t_stat, p_value = stats.ttest_1samp(unfav_arr, 0)
            patterns["unfavorable"] = SeasonalityPattern(
                pattern_type="sell_in_may",
                period_key="May-October (Unfavorable)",
                avg_return=float(np.mean(unfav_arr)),
                median_return=float(np.median(unfav_arr)),
                win_rate=float(np.mean(unfav_arr > 0)),
                avg_volatility=float(np.std(unfav_arr)),
                sample_size=len(unfav_arr),
                t_statistic=float(t_stat),
                p_value=float(p_value),
                is_significant=p_value < self.significance_level,
                historical_data=unfav_arr.tolist()
            )
        
        return patterns
    
    def analyze_turn_of_month(
        self,
        df: pd.DataFrame
    ) -> SeasonalityPattern:
        """
        Analyse l'effet 'Turn of Month'
        
        Derniers 3 jours + premiers 3 jours du mois
        """
        
        df = df.copy()
        df['date'] = pd.to_datetime(df.index if isinstance(df.index, pd.DatetimeIndex) else df['date'])
        df['day'] = df['date'].dt.day
        df['days_in_month'] = df['date'].dt.days_in_month
        
        # Identifier les jours TOM
        df['is_tom'] = (df['day'] <= 3) | (df['day'] >= df['days_in_month'] - 2)
        
        tom_returns = df[df['is_tom']]['daily_return'].values
        non_tom_returns = df[~df['is_tom']]['daily_return'].values
        
        if len(tom_returns) < self.min_samples:
            return None
        
        t_stat, p_value = stats.ttest_ind(tom_returns, non_tom_returns)
        
        return SeasonalityPattern(
            pattern_type="turn_of_month",
            period_key="Last 3 + First 3 days",
            avg_return=float(np.mean(tom_returns)),
            median_return=float(np.median(tom_returns)),
            win_rate=float(np.mean(tom_returns > 0)),
            avg_volatility=float(np.std(tom_returns)),
            sample_size=len(tom_returns),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            is_significant=p_value < self.significance_level,
            historical_data=tom_returns.tolist()
        )
    
    # ═══════════════════════════════════════════════════════════════
    # PRÉVISION COMPOSITE
    # ═══════════════════════════════════════════════════════════════
    
    def get_seasonality_forecast(
        self,
        df: pd.DataFrame,
        target_date: date,
        horizon_days: int = 21
    ) -> SeasonalityForecast:
        """
        Génère une prévision basée sur tous les patterns saisonniers
        
        Args:
            df: Données historiques
            target_date: Date de référence
            horizon_days: Horizon de prévision
            
        Returns:
            SeasonalityForecast avec score composite
        """
        
        # Analyser tous les patterns
        monthly = self.analyze_monthly_seasonality(df)
        presidential = self.analyze_presidential_cycle(df)
        santa = self.analyze_santa_rally(df)
        sell_may = self.analyze_sell_in_may(df)
        tom = self.analyze_turn_of_month(df)
        
        # Déterminer le contexte actuel
        current_month = target_date.month
        current_year = target_date.year
        cycle_year = ((current_year - 2020) % 4) + 1
        
        # Collecter les patterns pertinents
        relevant_patterns = []
        pattern_scores = []
        
        # Pattern mensuel
        if current_month in monthly:
            pattern = monthly[current_month]
            relevant_patterns.append(pattern)
            # Score pondéré par significativité
            weight = 1.0 if pattern.is_significant else 0.5
            pattern_scores.append(pattern.avg_return * weight)
        
        # Pattern présidentiel
        if cycle_year in presidential:
            pattern = presidential[cycle_year]
            relevant_patterns.append(pattern)
            weight = 0.8 if pattern.is_significant else 0.4
            # Ajuster pour la proportion de l'année écoulée
            year_progress = target_date.timetuple().tm_yday / 365
            remaining_factor = 1 - year_progress
            pattern_scores.append(pattern.avg_return * remaining_factor * weight)
        
        # Santa Rally (si applicable)
        if santa and current_month == 12 and target_date.day >= 20:
            relevant_patterns.append(santa)
            weight = 1.2 if santa.is_significant else 0.6
            pattern_scores.append(santa.avg_return * weight)
        
        # Sell in May context
        if sell_may:
            if current_month in [11, 12, 1, 2, 3, 4] and "favorable" in sell_may:
                pattern = sell_may["favorable"]
                weight = 0.5 if pattern.is_significant else 0.25
                pattern_scores.append(pattern.avg_return * weight / 6)  # Normaliser
                relevant_patterns.append(pattern)
            elif current_month in [5, 6, 7, 8, 9, 10] and "unfavorable" in sell_may:
                pattern = sell_may["unfavorable"]
                weight = 0.5 if pattern.is_significant else 0.25
                pattern_scores.append(pattern.avg_return * weight / 6)
                relevant_patterns.append(pattern)
        
        # Turn of month (si applicable)
        if tom and (target_date.day <= 3 or target_date.day >= 28):
            relevant_patterns.append(tom)
            pattern_scores.append(tom.avg_return * 0.3)
        
        # Calculer le score composite
        if pattern_scores:
            composite_score = np.mean(pattern_scores)
            # Ajuster pour l'horizon
            expected_return = composite_score * (horizon_days / 21)  # Normaliser à 1 mois
        else:
            composite_score = 0
            expected_return = 0
        
        # Calculer la confiance basée sur le nombre de patterns significatifs
        n_significant = sum(1 for p in relevant_patterns if p.is_significant)
        confidence = min(0.8, 0.3 + 0.15 * n_significant)
        
        # Collecter les returns historiques analogues
        analog_returns = []
        for pattern in relevant_patterns:
            analog_returns.extend(pattern.historical_data[-10:])
        
        return SeasonalityForecast(
            expected_return=float(expected_return),
            confidence=float(confidence),
            primary_patterns=relevant_patterns[:5],  # Top 5
            composite_score=float(composite_score),
            historical_analog_returns=analog_returns
        )
