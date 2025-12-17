"""
データベースからグラフを構築するモジュール
"""

import networkx as nx
from typing import List, Dict, Optional
import logging

from ..database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


class GraphBuilder:
    """データベースからNetworkXグラフを構築するクラス"""
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        初期化
        
        Args:
            db_connection: データベース接続オブジェクト
        """
        self.db = db_connection
    
    def build_graph(self, directed: bool = False, weight_attribute: Optional[str] = None) -> nx.Graph:
        """
        データベースからグラフを構築
        
        Args:
            directed: 有向グラフかどうか
            weight_attribute: エッジの重み属性名（Noneの場合は重みなし）
        
        Returns:
            NetworkXグラフオブジェクト
        """
        if directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        
        try:
            # ノードを取得して追加
            logger.info("ノードデータを取得しています...")
            nodes = self.db.get_nodes()
            logger.info(f"{len(nodes)}個のノードを取得しました")
            
            for node in nodes:
                node_id = node.get('id') or node.get('node_id') or node.get('pubkey')
                if node_id:
                    # ノードの属性を追加
                    attrs = {k: v for k, v in node.items() if k not in ['id', 'node_id', 'pubkey']}
                    graph.add_node(node_id, **attrs)
            
            # エッジを取得して追加
            logger.info("エッジデータを取得しています...")
            edges = self.db.get_edges()
            logger.info(f"{len(edges)}個のエッジを取得しました")
            
            for edge in edges:
                # エッジの始点と終点を取得（カラム名はデータベース構造に応じて調整が必要）
                source = edge.get('source') or edge.get('from_node') or edge.get('node1')
                target = edge.get('target') or edge.get('to_node') or edge.get('node2')
                
                if source and target:
                    edge_attrs = {k: v for k, v in edge.items() 
                                 if k not in ['source', 'target', 'from_node', 'to_node', 'node1', 'node2']}
                    
                    # 重み属性がある場合は追加
                    if weight_attribute and weight_attribute in edge_attrs:
                        graph.add_edge(source, target, weight=edge_attrs[weight_attribute], **edge_attrs)
                    else:
                        graph.add_edge(source, target, **edge_attrs)
            
            logger.info(f"グラフを構築しました（ノード数: {graph.number_of_nodes()}, エッジ数: {graph.number_of_edges()}）")
            return graph
        
        except Exception as e:
            logger.error(f"グラフの構築中にエラーが発生しました: {e}")
            raise

