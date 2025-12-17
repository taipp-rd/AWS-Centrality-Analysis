"""
中心性計算を統合管理するクラス
"""

import networkx as nx
import yaml
import os
from typing import Dict, Optional, List
import logging

from .betweenness import BetweennessCentrality
from .closeness import ClosenessCentrality
from .eigenvector import EigenvectorCentrality

logger = logging.getLogger(__name__)


class CentralityCalculator:
    """複数の中心性指標を計算・管理するクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
                'config.yaml'
            )
        
        self.config = self._load_config(config_path)
        self._initialize_calculators()
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('centrality', {})
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗しました: {e}。デフォルト設定を使用します。")
            return {}
    
    def _initialize_calculators(self):
        """各中心性計算クラスを初期化"""
        betweenness_config = self.config.get('betweenness', {})
        closeness_config = self.config.get('closeness', {})
        eigenvector_config = self.config.get('eigenvector', {})
        
        self.betweenness = BetweennessCentrality(
            normalized=betweenness_config.get('normalized', True),
            directed=betweenness_config.get('directed', False),
            weight=betweenness_config.get('weight')
        )
        
        self.closeness = ClosenessCentrality(
            normalized=closeness_config.get('normalized', True),
            distance=closeness_config.get('distance'),
            wf_improved=closeness_config.get('wf_improved', True)
        )
        
        self.eigenvector = EigenvectorCentrality(
            max_iter=eigenvector_config.get('max_iter', 100),
            tol=eigenvector_config.get('tol', 1.0e-6),
            weight=eigenvector_config.get('weight')
        )
    
    def calculate_all(self, graph: nx.Graph, transaction_size_msat: Optional[int] = None) -> Dict[str, Dict]:
        """
        すべての中心性指標を計算
        
        Args:
            graph: NetworkXグラフオブジェクト
            transaction_size_msat: トランザクションサイズ（millisatoshi）。指定された場合、ルーティング手数料を重みとして計算。
        
        Returns:
            中心性指標名をキー、中心性スコア辞書を値とする辞書
        """
        results = {}
        
        try:
            logger.info("媒介中心性の計算を開始します...")
            results['betweenness'] = self.betweenness.calculate(graph, transaction_size_msat=transaction_size_msat)
        except Exception as e:
            logger.error(f"媒介中心性の計算に失敗しました: {e}")
            results['betweenness'] = {}
        
        try:
            logger.info("近接中心性の計算を開始します...")
            results['closeness'] = self.closeness.calculate(graph, transaction_size_msat=transaction_size_msat)
        except Exception as e:
            logger.error(f"近接中心性の計算に失敗しました: {e}")
            results['closeness'] = {}
        
        try:
            logger.info("近似中心性の計算を開始します...")
            results['eigenvector'] = self.eigenvector.calculate(graph, transaction_size_msat=transaction_size_msat)
        except Exception as e:
            logger.error(f"近似中心性の計算に失敗しました: {e}")
            results['eigenvector'] = {}
        
        return results
    
    def calculate_betweenness(self, graph: nx.Graph, transaction_size_msat: Optional[int] = None) -> Dict:
        """媒介中心性のみを計算"""
        return self.betweenness.calculate(graph, transaction_size_msat=transaction_size_msat)
    
    def calculate_closeness(self, graph: nx.Graph, transaction_size_msat: Optional[int] = None) -> Dict:
        """近接中心性のみを計算"""
        return self.closeness.calculate(graph, transaction_size_msat=transaction_size_msat)
    
    def calculate_eigenvector(self, graph: nx.Graph, transaction_size_msat: Optional[int] = None) -> Dict:
        """近似中心性のみを計算"""
        return self.eigenvector.calculate(graph, transaction_size_msat=transaction_size_msat)
