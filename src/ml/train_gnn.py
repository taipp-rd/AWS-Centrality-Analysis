"""
グラフニューラルネットワーク（GNN）モデルの訓練スクリプト（SageMaker用）
"""

import argparse
import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import dgl
import dgl.nn as dglnn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCNModel(nn.Module):
    """Graph Convolutional Network (GCN) モデル"""
    
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, dropout=0.5):
        super(GCNModel, self).__init__()
        self.layers = nn.ModuleList()
        
        # 入力層
        self.layers.append(dglnn.GraphConv(in_feats, hidden_feats, activation=torch.relu))
        
        # 隠れ層
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GraphConv(hidden_feats, hidden_feats, activation=torch.relu))
        
        # 出力層
        self.layers.append(dglnn.GraphConv(hidden_feats, out_feats))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


class GATModel(nn.Module):
    """Graph Attention Network (GAT) モデル"""
    
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers=2, num_heads=8, dropout=0.5):
        super(GATModel, self).__init__()
        self.layers = nn.ModuleList()
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 入力層
        self.layers.append(dglnn.GATConv(in_feats, hidden_feats, num_heads=num_heads, activation=torch.relu, dropout=dropout))
        
        # 隠れ層
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.GATConv(hidden_feats * num_heads, hidden_feats, num_heads=num_heads, activation=torch.relu, dropout=dropout))
        
        # 出力層（num_heads=1で次元を削減）
        if num_layers > 1:
            self.layers.append(dglnn.GATConv(hidden_feats * num_heads, out_feats, num_heads=1, activation=None))
        else:
            # 1層のみの場合
            self.layers.append(dglnn.GATConv(in_feats, out_feats, num_heads=1, activation=None))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
            # マルチヘッドアテンションの場合、最後の次元を結合
            # 出力層（num_heads=1）の場合は形状が(N, out_feats)になる
            if len(h.shape) == 3:
                h = h.flatten(1)
        return h


def load_graph_data(data_path):
    """
    グラフデータを読み込む
    
    DGLのload_graphsは(graph_list, labels_dict)のタプルを返す
    """
    logger.info(f"グラフデータを読み込んでいます: {data_path}")
    
    # データファイルを検索
    files = [f for f in os.listdir(data_path) if f.endswith('.pkl') or f.endswith('.bin')]
    
    if not files:
        raise ValueError(f"グラフデータファイルが見つかりません: {data_path}")
    
    # DGLグラフを読み込む
    # dgl.load_graphs()は(graph_list, labels_dict)を返す
    graph_file = os.path.join(data_path, files[0])
    graph_list, labels_dict = dgl.load_graphs(graph_file)
    graph = graph_list[0]  # 最初のグラフを取得
    
    logger.info(f"グラフ情報: ノード数={graph.number_of_nodes()}, エッジ数={graph.number_of_edges()}")
    logger.info(f"ラベル辞書のキー: {list(labels_dict.keys()) if labels_dict else 'なし'}")
    
    return graph, labels_dict


def train_model(model, graph, features, labels, train_mask, val_mask, 
                epochs, lr, device):
    """
    モデルを訓練
    
    理論的改善点:
    1. 勾配クリッピングを追加（勾配爆発の防止）
    2. 学習率スケジューラーを追加
    3. より詳細な評価指標を追加
    """
    model = model.to(device)
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.MSELoss()
    
    # 学習率スケジューラー（オプション）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # フォワードパス
        logits = model(graph, features)
        
        # 形状の確認と修正
        if logits.shape != labels.shape:
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(1)
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, :labels.shape[1]]
        
        # 訓練損失
        train_loss = criterion(logits[train_mask], labels[train_mask])
        
        # バックワードパス
        train_loss.backward()
        
        # 勾配クリッピング（勾配爆発の防止）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 検証
        model.eval()
        with torch.no_grad():
            val_logits = model(graph, features)
            
            # 形状の確認と修正
            if val_logits.shape != labels.shape:
                if len(val_logits.shape) == 1:
                    val_logits = val_logits.unsqueeze(1)
                if val_logits.shape[1] != labels.shape[1]:
                    val_logits = val_logits[:, :labels.shape[1]]
            
            val_loss = criterion(val_logits[val_mask], labels[val_mask])
            
            # 追加の評価指標（MAE）
            val_mae = nn.L1Loss()(val_logits[val_mask], labels[val_mask])
        
        # 学習率スケジューラーの更新
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: Train Loss={train_loss.item():.6f}, Val Loss={val_loss.item():.6f}, Val MAE={val_mae.item():.6f}")
        
        # 早期停止
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 最良モデルの保存
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                # 最良モデルを復元
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    logger.info(f"訓練完了: 最良検証損失={best_val_loss.item():.6f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    
    # SageMaker環境変数
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/tmp'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/tmp'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', None))
    
    # ハイパーパラメータ
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-type', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--num-heads', type=int, default=8)  # GAT用
    parser.add_argument('--seed', type=int, default=42)  # 再現性のため
    
    args = parser.parse_args()
    
    # 再現性のためシードを設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用デバイス: {device}")
    logger.info(f"シード: {args.seed}")
    
    # データの読み込み
    graph, labels = load_graph_data(args.train)
    
    # ノード特徴量を取得（既にグラフに含まれている場合）
    if 'feat' in graph.ndata:
        features = graph.ndata['feat'].float()
        logger.info(f"グラフから特徴量を取得しました: {features.shape}")
    else:
        # デフォルト特徴量（次数など）
        # 注意: 中心性指標を特徴量として使用する場合は、ラベルとして使用しないこと（データリーク防止）
        in_degrees = graph.in_degrees().float()
        out_degrees = graph.out_degrees().float()
        features = torch.stack([in_degrees, out_degrees], dim=1)
        
        # 中心性指標を特徴量として追加する場合（オプション）
        # 注意: これらを特徴量として使用する場合、ラベルには使用しないこと
        if 'betweenness' in graph.ndata:
            features = torch.cat([features, graph.ndata['betweenness'].unsqueeze(1).float()], dim=1)
        if 'closeness' in graph.ndata:
            features = torch.cat([features, graph.ndata['closeness'].unsqueeze(1).float()], dim=1)
        if 'eigenvector' in graph.ndata:
            features = torch.cat([features, graph.ndata['eigenvector'].unsqueeze(1).float()], dim=1)
        
        logger.info(f"デフォルト特徴量を作成しました: {features.shape}")
    
    # 特徴量の正規化（重要: 異なるスケールの特徴量を統一）
    from torch.nn.functional import normalize
    # L2正規化（オプション: 必要に応じてコメントアウト）
    # features = normalize(features, p=2, dim=1)
    
    # または、Z-score正規化（推奨）
    feature_mean = features.mean(dim=0, keepdim=True)
    feature_std = features.std(dim=0, keepdim=True) + 1e-8  # ゼロ除算防止
    features = (features - feature_mean) / feature_std
    logger.info("特徴量をZ-score正規化しました")
    
    # ラベルを取得
    # ラベルは通常graph.ndata['label']に保存されている
    if 'label' in graph.ndata:
        labels = graph.ndata['label'].float()
        logger.info(f"グラフからラベルを取得しました: {labels.shape}")
    elif isinstance(labels_dict, dict) and len(labels_dict) > 0:
        # labels_dictから取得（通常は空だが、念のため）
        target_key = list(labels_dict.keys())[0]
        labels = labels_dict[target_key].float()
        logger.info(f"ラベル辞書からラベルを取得しました: {labels.shape}")
    else:
        # デフォルトラベル（実際のデータがない場合のみ）
        logger.warning("ラベルが見つかりません。デフォルトラベルを使用します。")
        labels = torch.randn(graph.number_of_nodes(), 1)
    
    # ラベルの形状を確認・修正
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    elif len(labels.shape) > 2:
        labels = labels.view(-1, 1)
    
    in_feats = features.shape[1]
    out_feats = labels.shape[1] if len(labels.shape) > 1 else 1
    
    # 訓練/検証マスクを作成（再現性のためシードを設定）
    num_nodes = graph.number_of_nodes()
    train_size = int(0.8 * num_nodes)
    
    # 再現性のためシードを設定
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    indices = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[indices[:train_size]] = True
    val_mask = ~train_mask
    
    logger.info(f"訓練データ: {train_mask.sum().item()}ノード, 検証データ: {val_mask.sum().item()}ノード")
    
    # モデルの作成
    if args.model_type == 'gcn':
        model = GCNModel(in_feats, args.hidden_dim, out_feats, 
                        args.num_layers, args.dropout)
    else:  # gat
        model = GATModel(in_feats, args.hidden_dim, out_feats,
                        args.num_layers, args.num_heads, args.dropout)
    
    logger.info(f"モデルタイプ: {args.model_type}")
    logger.info(f"入力特徴量数: {in_feats}, 出力次元: {out_feats}")
    
    # モデルの訓練
    model = train_model(model, graph, features, labels, train_mask, val_mask,
                       args.epochs, args.learning_rate, device)
    
    # モデルの保存
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"モデルを保存しました: {model_path}")
    
    # メタデータの保存
    metadata = {
        'model_type': args.model_type,
        'in_feats': in_feats,
        'hidden_dim': args.hidden_dim,
        'out_feats': out_feats,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'hyperparameters': vars(args)
    }
    
    metadata_path = os.path.join(args.model_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"メタデータを保存しました: {metadata_path}")
    
    # グラフ情報の保存（推論時に必要）
    graph_info = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'node_feature_dim': in_feats
    }
    
    graph_info_path = os.path.join(args.model_dir, 'graph_info.json')
    with open(graph_info_path, 'w') as f:
        json.dump(graph_info, f, indent=2)
    
    logger.info("訓練が完了しました")


if __name__ == '__main__':
    main()
