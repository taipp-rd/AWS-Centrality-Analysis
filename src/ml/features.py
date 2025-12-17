"""
特徴量抽出モジュール
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """グラフから機械学習用の特徴量を抽出するクラス"""
    
    def __init__(self, graph: nx.Graph, centrality_scores: Dict[str, Dict]):
        """
        初期化
        
        Args:
            graph: NetworkXグラフオブジェクト
            centrality_scores: 中心性スコアの辞書
        """
        self.graph = graph
        self.centrality_scores = centrality_scores
    
    def extract_node_features(self, node_id: str) -> Dict[str, float]:
        """
        ノードの特徴量を抽出
        
        Args:
            node_id: ノードID
        
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # 基本統計量
        features['degree'] = self.graph.degree(node_id)
        features['in_degree'] = self.graph.in_degree(node_id) if self.graph.is_directed() else self.graph.degree(node_id)
        features['out_degree'] = self.graph.out_degree(node_id) if self.graph.is_directed() else self.graph.degree(node_id)
        
        # 中心性スコア
        if 'betweenness' in self.centrality_scores:
            features['betweenness_centrality'] = self.centrality_scores['betweenness'].get(node_id, 0.0)
        
        if 'closeness' in self.centrality_scores:
            features['closeness_centrality'] = self.centrality_scores['closeness'].get(node_id, 0.0)
        
        if 'eigenvector' in self.centrality_scores:
            features['eigenvector_centrality'] = self.centrality_scores['eigenvector'].get(node_id, 0.0)
        
        # クラスタリング係数
        try:
            features['clustering'] = nx.clustering(self.graph, node_id)
        except:
            features['clustering'] = 0.0
        
        # 近傍ノードの平均次数
        neighbors = list(self.graph.neighbors(node_id))
        if neighbors:
            neighbor_degrees = [self.graph.degree(n) for n in neighbors]
            features['avg_neighbor_degree'] = np.mean(neighbor_degrees)
        else:
            features['avg_neighbor_degree'] = 0.0
        
        return features
    
    def extract_edge_features(self, source: str, target: str) -> Dict[str, float]:
        """
        エッジの特徴量を抽出
        
        Args:
            source: 始点ノードID
            target: 終点ノードID
        
        Returns:
            特徴量の辞書
        """
        features = {}
        
        # ノードの特徴量を取得
        source_features = self.extract_node_features(source)
        target_features = self.extract_node_features(target)
        
        # エッジ固有の特徴量
        features['source_degree'] = source_features['degree']
        features['target_degree'] = target_features['degree']
        features['source_betweenness'] = source_features.get('betweenness_centrality', 0.0)
        features['target_betweenness'] = target_features.get('betweenness_centrality', 0.0)
        features['source_closeness'] = source_features.get('closeness_centrality', 0.0)
        features['target_closeness'] = target_features.get('closeness_centrality', 0.0)
        features['source_eigenvector'] = source_features.get('eigenvector_centrality', 0.0)
        features['target_eigenvector'] = target_features.get('eigenvector_centrality', 0.0)
        
        # 共通近傍数
        try:
            common_neighbors = list(nx.common_neighbors(self.graph, source, target))
            features['common_neighbors'] = len(common_neighbors)
        except:
            features['common_neighbors'] = 0
        
        # 最短経路長（存在する場合）
        try:
            if nx.has_path(self.graph, source, target):
                features['shortest_path_length'] = nx.shortest_path_length(self.graph, source, target)
            else:
                features['shortest_path_length'] = -1
        except:
            features['shortest_path_length'] = -1
        
        return features
    
    def create_node_dataframe(self) -> pd.DataFrame:
        """
        すべてのノードの特徴量をDataFrameとして作成
        
        Returns:
            特徴量のDataFrame
        """
        node_features = []
        for node_id in self.graph.nodes():
            features = self.extract_node_features(node_id)
            features['node_id'] = node_id
            node_features.append(features)
        
        return pd.DataFrame(node_features)
    
    def create_edge_dataframe(self) -> pd.DataFrame:
        """
        すべてのエッジの特徴量をDataFrameとして作成
        
        Returns:
            特徴量のDataFrame
        """
        edge_features = []
        for source, target in self.graph.edges():
            features = self.extract_edge_features(source, target)
            features['source'] = source
            features['target'] = target
            edge_features.append(features)
        
        return pd.DataFrame(edge_features)

