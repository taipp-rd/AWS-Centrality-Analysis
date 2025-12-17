"""
媒介中心性 (Betweenness Centrality) 計算モジュール
"""

import networkx as nx
import numpy as np
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BetweennessCentrality:
    """媒介中心性を計算するクラス"""
    
    def __init__(self, normalized: bool = True, directed: bool = False, weight: Optional[str] = None):
        """
        初期化
        
        Args:
            normalized: 正規化するかどうか
            directed: 有向グラフかどうか
            weight: エッジの重み属性名（Noneの場合は重みなし）
        """
        self.normalized = normalized
        self.directed = directed
        self.weight = weight
    
    def calculate(self, graph: nx.Graph, k: Optional[int] = None, 
                  seed: Optional[int] = None) -> Dict[Union[str, int], float]:
        """
        媒介中心性を計算
        
        Args:
            graph: NetworkXグラフオブジェクト
            k: サンプリングするノード数（Noneの場合は全ノード、大規模グラフ用）
            seed: ランダムシード（kが指定されている場合）
        
        Returns:
            ノードIDをキー、中心性スコアを値とする辞書
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            
            logger.info(f"媒介中心性の計算を開始します（ノード数: {num_nodes}, エッジ数: {num_edges}）")
            
            if k and k < num_nodes:
                logger.info(f"サンプリングモード: 上位{k}ノードのみを計算します")
                # 次数の高いノードを優先的に選択
                degree_dict = dict(graph.degree())
                top_k_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:k]
                top_k_node_list = [node for node, _ in top_k_nodes]
                
                if self.weight:
                    centrality = nx.betweenness_centrality(
                        graph,
                        k=k,
                        normalized=self.normalized,
                        weight=self.weight,
                        seed=seed
                    )
                else:
                    centrality = nx.betweenness_centrality(
                        graph,
                        k=k,
                        normalized=self.normalized,
                        seed=seed
                    )
            else:
                if self.weight:
                    centrality = nx.betweenness_centrality(
                        graph,
                        normalized=self.normalized,
                        weight=self.weight
                    )
                else:
                    centrality = nx.betweenness_centrality(
                        graph,
                        normalized=self.normalized
                    )
            
            # 統計情報をログ出力
            if centrality:
                values = list(centrality.values())
                logger.info(f"媒介中心性の計算が完了しました（ノード数: {len(centrality)}）")
                logger.info(f"  平均: {np.mean(values):.6f}, 最大: {np.max(values):.6f}, 最小: {np.min(values):.6f}")
            
            return centrality
        
        except Exception as e:
            logger.error(f"媒介中心性の計算中にエラーが発生しました: {e}")
            logger.error(f"グラフ情報: ノード数={graph.number_of_nodes()}, エッジ数={graph.number_of_edges()}")
            raise
    
    def calculate_edge(self, graph: nx.Graph) -> Dict[tuple, float]:
        """
        エッジの媒介中心性を計算
        
        Args:
            graph: NetworkXグラフオブジェクト
        
        Returns:
            エッジ（タプル）をキー、中心性スコアを値とする辞書
        """
        try:
            if self.weight:
                edge_centrality = nx.edge_betweenness_centrality(
                    graph,
                    normalized=self.normalized,
                    weight=self.weight
                )
            else:
                edge_centrality = nx.edge_betweenness_centrality(
                    graph,
                    normalized=self.normalized
                )
            
            logger.info(f"エッジ媒介中心性の計算が完了しました（エッジ数: {len(edge_centrality)}）")
            return edge_centrality
        
        except Exception as e:
            logger.error(f"エッジ媒介中心性の計算中にエラーが発生しました: {e}")
            raise

