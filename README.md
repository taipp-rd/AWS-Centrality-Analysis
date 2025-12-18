# ライトニングネットワーク中心性分析プロジェクト

ビットコインのライトニングネットワークにおける3つの中心性指標（媒介中心性、近接中心性、近似中心性）を分析し、ルーティング可能性を評価するプロジェクトです。

## 機能

- **中心性計算**: 3つの中心性指標（媒介、近接、近似）を計算
- **グラフニューラルネットワーク（GNN）**: GCN/GATによる中心性予測とルーティング可能性分析
- **グラフ分析**: ライトニングネットワークの構造を分析
- **ルーティング分析**: どのノードとチャネルを開設すべきかを推薦
- **AWS SageMaker統合**: GNNモデルの訓練とデプロイ
- **可視化**: 中心性分布とグラフ構造の可視化

## プロジェクト構造

```
.
├── config/
│   └── config.yaml          # 設定ファイル
├── src/
│   ├── database/            # データベース接続モジュール
│   │   ├── __init__.py
│   │   └── connection.py
│   ├── graph/               # グラフ構築モジュール
│   │   ├── __init__.py
│   │   └── builder.py
│   ├── centrality/          # 中心性計算モジュール
│   │   ├── __init__.py
│   │   ├── betweenness.py   # 媒介中心性
│   │   ├── closeness.py      # 近接中心性
│   │   ├── eigenvector.py    # 近似中心性
│   │   └── calculator.py    # 統合計算器
│   ├── ml/                  # グラフニューラルネットワークモジュール
│   │   ├── __init__.py
│   │   ├── features.py      # 特徴量抽出
│   │   ├── pipeline.py      # SageMaker GNNパイプライン
│   │   └── train_gnn.py     # GNN訓練スクリプト
│   └── analysis/             # 分析・可視化モジュール
│       ├── __init__.py
│       ├── visualizer.py     # 可視化
│       └── analyzer.py       # ルーティング分析
├── notebooks/
│   └── lightning_network_centrality_analysis.ipynb  # メインのJupyter Notebook
├── results/                  # 分析結果の保存先
├── logs/                     # ログファイル
├── requirements.txt          # Python依存パッケージ
└── README.md                 # このファイル
```

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. 設定ファイルの編集

`config/config.yaml`を編集して、データベース接続情報やAWS設定を確認してください。

### 3. AWS SageMakerの設定（GNN訓練用）

GNNモデルの訓練を使用する場合、以下の設定が必要です：

1. **IAMロール**: SageMakerの実行ロールを作成（S3アクセス権限を含む）
2. **S3バケット**: データとモデルの保存用バケットを作成
3. `config/config.yaml`でこれらの情報を設定
4. 詳細は [`docs/AWS_SAGEMAKER_GUIDE.md`](docs/AWS_SAGEMAKER_GUIDE.md) を参照

## 使用方法

### AWS SageMaker Studio Lab / Studio Classicでの実行

1. AWS SageMaker Studio LabまたはStudio Classicにログイン
2. このプロジェクトをアップロードまたはクローン
3. `notebooks/lightning_network_centrality_analysis.ipynb`を開く
4. セルを順番に実行

### ローカルでの実行

```bash
# Jupyter Notebookを起動
jupyter notebook

# または JupyterLabを起動
jupyter lab
```

その後、`notebooks/lightning_network_centrality_analysis.ipynb`を開いて実行します。

## 中心性指標について

### 1. 媒介中心性 (Betweenness Centrality)

ノードが他のノード間の最短経路にどれだけ頻繁に現れるかを測定します。高い値を持つノードは、ネットワーク内で重要な「橋渡し」の役割を果たします。

### 2. 近接中心性 (Closeness Centrality)

ノードから他のすべてのノードへの平均距離の逆数です。高い値を持つノードは、ネットワーク全体に素早くアクセスできる位置にあります。

### 3. 近似中心性 (Eigenvector Centrality)

重要なノードに接続されているノードが重要とされる指標です。PageRankアルゴリズムの基盤となっています。

## 拡張性

このプロジェクトは拡張可能な設計になっています：

- **新しい中心性指標の追加**: `src/centrality/`に新しいクラスを追加
- **新しい特徴量の追加**: `src/ml/features.py`の`FeatureExtractor`クラスを拡張
- **新しい分析手法の追加**: `src/analysis/`に新しいモジュールを追加
- **設定の追加**: `config/config.yaml`に新しい設定セクションを追加

## 出力ファイル

分析結果は`results/`ディレクトリに保存されます：

- `centrality_distribution.png`: 中心性分布のヒストグラム
- `graph_*.png`: 各中心性タイプでのグラフ可視化
- `top_nodes_comparison.png`: 上位ノードの比較
- `recommended_nodes.csv`: 推奨ノードのリスト
- `recommended_channels.csv`: 推奨チャネルのリスト
- `node_features.csv`: ノード特徴量
- `edge_features.csv`: エッジ特徴量

## 注意事項

- データベース接続情報は機密情報です。設定ファイルをバージョン管理に含めないでください
- 大規模なグラフでは計算に時間がかかる場合があります
- AWS SageMakerを使用する場合は、適切なIAM権限が必要です


## 参考文献

- NetworkX Documentation: https://networkx.org/
- Deep Graph Library (DGL): https://www.dgl.ai/
- AWS SageMaker Documentation: https://docs.aws.amazon.com/sagemaker/
