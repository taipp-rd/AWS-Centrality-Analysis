"""
グラフ可視化モジュール
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, Optional, List
import os
import logging

logger = logging.getLogger(__name__)


class GraphVisualizer:
    """グラフと中心性の可視化を行うクラス"""
    
    def __init__(self, output_dir: str = './results'):
        """
        初期化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_centrality_distribution(self, centrality_scores: Dict[str, Dict], 
                                    save_path: Optional[str] = None):
        """
        中心性スコアの分布を可視化
        
        Args:
            centrality_scores: 中心性スコアの辞書
            save_path: 保存パス
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        centrality_names = ['betweenness', 'closeness', 'eigenvector']
        titles = ['媒介中心性', '近接中心性', '近似中心性']
        
        for idx, (name, title) in enumerate(zip(centrality_names, titles)):
            if name in centrality_scores:
                scores = list(centrality_scores[name].values())
                axes[idx].hist(scores, bins=50, edgecolor='black', alpha=0.7)
                axes[idx].set_title(title)
                axes[idx].set_xlabel('中心性スコア')
                axes[idx].set_ylabel('頻度')
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"中心性分布を保存しました: {save_path}")
        else:
            save_path = os.path.join(self.output_dir, 'centrality_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"中心性分布を保存しました: {save_path}")
        
        plt.close()
    
    def plot_graph_with_centrality(self, graph: nx.Graph, 
                                   centrality_scores: Dict[str, Dict],
                                   centrality_type: str = 'betweenness',
                                   top_n: int = 100,
                                   save_path: Optional[str] = None):
        """
        中心性に基づいてグラフを可視化
        
        Args:
            graph: NetworkXグラフオブジェクト
            centrality_scores: 中心性スコアの辞書
            centrality_type: 使用する中心性タイプ
            top_n: 表示する上位ノード数
            save_path: 保存パス
        """
        if centrality_type not in centrality_scores:
            logger.error(f"中心性タイプ '{centrality_type}' が見つかりません")
            return
        
        # 上位Nノードを取得
        scores = centrality_scores[centrality_type]
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_node_ids = [node_id for node_id, _ in sorted_nodes]
        
        # サブグラフを作成
        subgraph = graph.subgraph(top_node_ids)
        
        # レイアウトを計算
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # ノードのサイズを中心性に基づいて設定
        node_sizes = [scores.get(node, 0) * 1000 for node in subgraph.nodes()]
        
        plt.figure(figsize=(16, 12))
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color=node_sizes, cmap=plt.cm.viridis, 
                              alpha=0.7)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5)
        
        # ラベルは上位10ノードのみ表示
        top_10_nodes = top_node_ids[:10]
        labels = {node: node[:8] + '...' if len(str(node)) > 8 else str(node) 
                 for node in top_10_nodes}
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title(f'グラフ可視化 - {centrality_type}中心性（上位{top_n}ノード）')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, f'graph_{centrality_type}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"グラフ可視化を保存しました: {save_path}")
        plt.close()
    
    def plot_top_nodes_comparison(self, centrality_scores: Dict[str, Dict],
                                  top_n: int = 20,
                                  save_path: Optional[str] = None):
        """
        各中心性指標の上位ノードを比較
        
        Args:
            centrality_scores: 中心性スコアの辞書
            top_n: 表示する上位ノード数
            save_path: 保存パス
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        centrality_names = ['betweenness', 'closeness', 'eigenvector']
        titles = ['媒介中心性', '近接中心性', '近似中心性']
        
        for idx, (name, title) in enumerate(zip(centrality_names, titles)):
            if name in centrality_scores:
                scores = centrality_scores[name]
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                nodes = [str(item[0])[:15] + '...' if len(str(item[0])) > 15 else str(item[0]) 
                        for item in sorted_items]
                values = [item[1] for item in sorted_items]
                
                axes[idx].barh(range(len(nodes)), values)
                axes[idx].set_yticks(range(len(nodes)))
                axes[idx].set_yticklabels(nodes, fontsize=8)
                axes[idx].set_xlabel('中心性スコア')
                axes[idx].set_title(f'{title} - 上位{top_n}ノード')
                axes[idx].invert_yaxis()
                axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.output_dir, 'top_nodes_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"上位ノード比較を保存しました: {save_path}")
        plt.close()

