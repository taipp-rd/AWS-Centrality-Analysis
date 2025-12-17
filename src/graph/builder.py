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
            # エッジを先に取得して、実際にチャネルを持つノードのみを取得
            logger.info("エッジデータを取得しています...")
            edges = self.db.get_edges(active_only=True, exclude_closed=True)  # 有効で閉じられていないチャネルのみ取得
            logger.info(f"{len(edges)}個のエッジを取得しました")
            
            # エッジからノードIDを抽出
            node_ids_from_edges = set()
            for edge in edges:
                source = edge.get('advertising_nodeid')
                target = edge.get('connecting_nodeid')
                if source:
                    node_ids_from_edges.add(source)
                if target:
                    node_ids_from_edges.add(target)
            
            # アクティブなチャネルを持つノードのみを取得
            logger.info("ノードデータを取得しています...")
            nodes = self.db.get_nodes(active_channels_only=True)
            logger.info(f"{len(nodes)}個のノードを取得しました")
            
            # エッジに含まれるノードのみを追加
            for node in nodes:
                node_id = node.get('node_id')
                if node_id and node_id in node_ids_from_edges:
                    # ノードの属性を追加
                    attrs = {k: v for k, v in node.items() if k != 'node_id'}
                    graph.add_node(node_id, **attrs)
            
            for edge in edges:
                # channel_updateテーブルのadvertising_nodeidとconnecting_nodeidカラムを使用
                source = edge.get('advertising_nodeid')
                target = edge.get('connecting_nodeid')
                
                if source and target:
                    edge_attrs = {k: v for k, v in edge.items() 
                                 if k not in ['source', 'target', 'from_node', 'to_node', 'node1', 'node2', 
                                             'advertising_nodeid', 'connecting_nodeid']}
                    
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
