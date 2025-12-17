"""
近接中心性 (Closeness Centrality) 計算モジュール
"""

import networkx as nx
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ClosenessCentrality:
    """近接中心性を計算するクラス"""
    
    def __init__(self, normalized: bool = True, distance: Optional[str] = None, wf_improved: bool = True):
        """
        初期化
        
        Args:
            normalized: 正規化するかどうか
            distance: 距離属性名（Noneの場合は重みなし）
            wf_improved: Wasserman and Faust改善版を使用するかどうか
        """
        self.normalized = normalized
        self.distance = distance
        self.wf_improved = wf_improved
    
    def calculate(self, graph: nx.Graph, u: Optional[Union[str, int]] = None) -> Dict[Union[str, int], float]:
        """
        近接中心性を計算
        
        Args:
            graph: NetworkXグラフオブジェクト
            u: 特定のノードのみを計算する場合（Noneの場合は全ノード）
        
        Returns:
            ノードIDをキー、中心性スコアを値とする辞書
        """
        try:
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            
            logger.info(f"近接中心性の計算を開始します（ノード数: {num_nodes}, エッジ数: {num_edges}）")
            
            # 非連結グラフのチェック
            if not nx.is_connected(graph):
                num_components = nx.number_connected_components(graph)
                logger.warning(f"グラフは非連結です（連結成分数: {num_components}）。wf_improved=Trueが推奨されます。")
            
            if u is not None:
                # 単一ノードのみ計算
                if self.distance:
                    score = nx.closeness_centrality(
                        graph,
                        u=u,
                        distance=self.distance,
                        wf_improved=self.wf_improved
                    )
                    centrality = {u: score}
                else:
                    score = nx.closeness_centrality(
                        graph,
                        u=u,
                        wf_improved=self.wf_improved
                    )
                    centrality = {u: score}
            else:
                if self.distance:
                    centrality = nx.closeness_centrality(
                        graph,
                        distance=self.distance,
                        wf_improved=self.wf_improved
                    )
                else:
                    centrality = nx.closeness_centrality(
                        graph,
                        wf_improved=self.wf_improved
                    )
            
            # 統計情報をログ出力
            if centrality:
                values = list(centrality.values())
                import numpy as np
                logger.info(f"近接中心性の計算が完了しました（ノード数: {len(centrality)}）")
                logger.info(f"  平均: {np.mean(values):.6f}, 最大: {np.max(values):.6f}, 最小: {np.min(values):.6f}")
            
            return centrality
        
        except Exception as e:
            logger.error(f"近接中心性の計算中にエラーが発生しました: {e}")
            logger.error(f"グラフ情報: ノード数={graph.number_of_nodes()}, エッジ数={graph.number_of_edges()}")
            raise

