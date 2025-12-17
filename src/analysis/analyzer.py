"""
ルーティング分析モジュール
"""

import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RoutingAnalyzer:
    """ルーティング可能性を分析するクラス"""
    
    def __init__(self, graph: nx.Graph, centrality_scores: Dict[str, Dict]):
        """
        初期化
        
        Args:
            graph: NetworkXグラフオブジェクト
            centrality_scores: 中心性スコアの辞書
        """
        self.graph = graph
        self.centrality_scores = centrality_scores
    
    def analyze_node_routing_potential(self, node_id: str) -> Dict[str, float]:
        """
        ノードのルーティング可能性を分析
        
        Args:
            node_id: ノードID
        
        Returns:
            分析結果の辞書
        """
        if node_id not in self.graph:
            logger.warning(f"ノード '{node_id}' がグラフに存在しません")
            return {}
        
        result = {
            'node_id': node_id,
            'degree': self.graph.degree(node_id),
        }
        
        # 各中心性スコアを追加
        for centrality_type, scores in self.centrality_scores.items():
            result[f'{centrality_type}_centrality'] = scores.get(node_id, 0.0)
        
        # ルーティング可能性スコア（複合指標）
        routing_score = (
            result.get('betweenness_centrality', 0.0) * 0.5 +
            result.get('closeness_centrality', 0.0) * 0.3 +
            result.get('eigenvector_centrality', 0.0) * 0.2
        )
        result['routing_potential'] = routing_score
        
        return result
    
    def recommend_nodes_for_channel(self, top_n: int = 50) -> pd.DataFrame:
        """
        チャネル開設に推奨されるノードを推薦
        
        Args:
            top_n: 推薦するノード数
        
        Returns:
            推薦ノードのDataFrame
        """
        recommendations = []
        
        for node_id in self.graph.nodes():
            analysis = self.analyze_node_routing_potential(node_id)
            recommendations.append(analysis)
        
        df = pd.DataFrame(recommendations)
        df = df.sort_values('routing_potential', ascending=False).head(top_n)
        
        logger.info(f"上位{top_n}ノードの推薦を完了しました")
        return df
    
    def analyze_channel_potential(self, source: str, target: str) -> Dict[str, float]:
        """
        特定のチャネルのルーティング可能性を分析
        
        Args:
            source: 始点ノードID
            target: 終点ノードID
        
        Returns:
            分析結果の辞書
        """
        if source not in self.graph or target not in self.graph:
            logger.warning(f"ノード '{source}' または '{target}' がグラフに存在しません")
            return {}
        
        result = {
            'source': source,
            'target': target,
            'existing_channel': self.graph.has_edge(source, target)
        }
        
        # 各ノードの中心性
        source_analysis = self.analyze_node_routing_potential(source)
        target_analysis = self.analyze_node_routing_potential(target)
        
        result['source_routing_potential'] = source_analysis.get('routing_potential', 0.0)
        result['target_routing_potential'] = target_analysis.get('routing_potential', 0.0)
        result['combined_potential'] = (
            result['source_routing_potential'] + result['target_routing_potential']
        ) / 2
        
        # 最短経路情報
        try:
            if nx.has_path(self.graph, source, target):
                result['shortest_path_length'] = nx.shortest_path_length(self.graph, source, target)
                result['shortest_paths_count'] = len(list(nx.all_shortest_paths(self.graph, source, target)))
            else:
                result['shortest_path_length'] = -1
                result['shortest_paths_count'] = 0
        except:
            result['shortest_path_length'] = -1
            result['shortest_paths_count'] = 0
        
        return result
    
    def recommend_channels(self, candidate_nodes: Optional[List[str]] = None,
                          top_n: int = 100) -> pd.DataFrame:
        """
        チャネル開設を推奨するエッジを推薦
        
        Args:
            candidate_nodes: 候補ノードのリスト（Noneの場合は全ノード）
            top_n: 推薦するチャネル数
        
        Returns:
            推薦チャネルのDataFrame
        """
        if candidate_nodes is None:
            candidate_nodes = list(self.graph.nodes())
        
        recommendations = []
        
        # 既存のエッジを除外
        existing_edges = set(self.graph.edges())
        
        # 候補ノードのペアを生成
        for i, source in enumerate(candidate_nodes):
            for target in candidate_nodes[i+1:]:
                if (source, target) not in existing_edges and (target, source) not in existing_edges:
                    analysis = self.analyze_channel_potential(source, target)
                    recommendations.append(analysis)
        
        df = pd.DataFrame(recommendations)
        if not df.empty:
            df = df.sort_values('combined_potential', ascending=False).head(top_n)
        
        logger.info(f"上位{top_n}チャネルの推薦を完了しました")
        return df

