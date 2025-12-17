"""
近似中心性 (Eigenvector Centrality) 計算モジュール
"""

import networkx as nx
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class EigenvectorCentrality:
    """近似中心性（固有ベクトル中心性）を計算するクラス"""
    
    def __init__(self, max_iter: int = 100, tol: float = 1.0e-6, weight: Optional[str] = None):
        """
        初期化
        
        Args:
            max_iter: 最大反復回数
            tol: 収束判定の許容誤差
            weight: エッジの重み属性名（Noneの場合は重みなし）
        """
        self.max_iter = max_iter
        self.tol = tol
        self.weight = weight
    
    def calculate(self, graph: nx.Graph) -> Dict[Union[str, int], float]:
        """
        近似中心性を計算
        
        Args:
            graph: NetworkXグラフオブジェクト
        
        Returns:
            ノードIDをキー、中心性スコアを値とする辞書
        """
        try:
            if self.weight:
                centrality = nx.eigenvector_centrality(
                    graph,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    weight=self.weight
                )
            else:
                centrality = nx.eigenvector_centrality(
                    graph,
                    max_iter=self.max_iter,
                    tol=self.tol
                )
            
            logger.info(f"近似中心性の計算が完了しました（ノード数: {len(centrality)}）")
            return centrality
        
        except Exception as e:
            logger.error(f"近似中心性の計算中にエラーが発生しました: {e}")
            raise

