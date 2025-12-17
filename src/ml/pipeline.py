"""
AWS SageMakerグラフニューラルネットワーク（GNN）パイプライン
"""

import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch
import pandas as pd
import numpy as np
import os
import yaml
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class GNNPipeline:
    """AWS SageMakerを使用したグラフニューラルネットワーク（GNN）パイプライン"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
                'config.yaml'
            )
        
        self.config = self._load_config(config_path)
        self._initialize_sagemaker()
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
            raise
    
    def _initialize_sagemaker(self):
        """SageMakerセッションを初期化"""
        try:
            aws_config = self.config.get('aws', {}).get('sagemaker', {})
            self.region = aws_config.get('region', 'ap-northeast-1')
            
            # IAMロールの取得（SageMaker内で実行する場合は自動取得）
            try:
                self.role = get_execution_role()
            except:
                self.role = aws_config.get('role')
                if not self.role:
                    logger.warning("IAMロールが設定されていません。手動で設定してください。")
            
            self.bucket = aws_config.get('bucket')
            if not self.bucket:
                logger.warning("S3バケットが設定されていません。手動で設定してください。")
            
            self.instance_type = aws_config.get('instance_type', 'ml.t3.medium')
            
            self.sagemaker_session = sagemaker.Session()
            logger.info("SageMakerセッションを初期化しました")
        
        except Exception as e:
            logger.error(f"SageMakerの初期化に失敗しました: {e}")
            raise
    
    def prepare_training_data(self, features_df: pd.DataFrame, target_column: str, 
                             output_path: str) -> str:
        """
        訓練データを準備してS3にアップロード
        
        Args:
            features_df: 特徴量DataFrame
            target_column: ターゲット列名
            output_path: S3出力パス
        
        Returns:
            S3のデータパス
        """
        try:
            # ターゲット列が存在することを確認
            if target_column not in features_df.columns:
                raise ValueError(f"ターゲット列 '{target_column}' がDataFrameに存在しません")
            
            # CSV形式で保存
            local_path = '/tmp/training_data.csv'
            features_df.to_csv(local_path, index=False)
            
            # S3にアップロード
            s3_path = f"s3://{self.bucket}/{output_path}/training_data.csv"
            self.sagemaker_session.upload_data(
                path=local_path,
                bucket=self.bucket,
                key_prefix=output_path
            )
            
            logger.info(f"訓練データをS3にアップロードしました: {s3_path}")
            return s3_path
        
        except Exception as e:
            logger.error(f"訓練データの準備中にエラーが発生しました: {e}")
            raise
    
    def train_gnn_model(self, train_data_path: str,
                       validation_data_path: Optional[str] = None,
                       hyperparameters: Optional[Dict] = None,
                       use_dgl_container: bool = True) -> Any:
        """
        GNNモデルを訓練（PyTorch + DGL）
        
        Args:
            train_data_path: 訓練データのS3パス
            validation_data_path: 検証データのS3パス（オプション）
            hyperparameters: ハイパーパラメータ
            use_dgl_container: DGLコンテナを使用するかどうか
        
        Returns:
            訓練済みエスティメーター
        """
        try:
            ml_config = self.config.get('ml', {}).get('gnn', {})
            
            if hyperparameters is None:
                hyperparameters = ml_config.get('hyperparameters', {
                    'hidden-dim': 64,
                    'num-layers': 2,
                    'dropout': 0.5,
                    'learning-rate': 0.01,
                    'epochs': 100,
                    'batch-size': 32
                })
            
            # 訓練スクリプトのパスを取得
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'src', 'ml', 'train_gnn.py'
            )
            
            # PyTorchエスティメーターを作成
            if use_dgl_container:
                # DGLコンテナを使用（ECRから取得）
                image_uri = self._get_dgl_container_uri()
                estimator = PyTorch(
                    entry_point=script_path,
                    role=self.role,
                    instance_type=self.instance_type,
                    instance_count=1,
                    image_uri=image_uri,
                    hyperparameters=hyperparameters,
                    output_path=f"s3://{self.bucket}/{self.config.get('aws', {}).get('s3', {}).get('prefix', 'models/')}",
                    sagemaker_session=self.sagemaker_session,
                    py_version='py3',
                    framework_version=None  # カスタムコンテナのため
                )
            else:
                # 標準PyTorchコンテナを使用（DGLをインストール）
                estimator = PyTorch(
                    entry_point=script_path,
                    role=self.role,
                    instance_type=self.instance_type,
                    instance_count=1,
                    framework_version='2.0.0',
                    py_version='py310',
                    hyperparameters=hyperparameters,
                    output_path=f"s3://{self.bucket}/{self.config.get('aws', {}).get('s3', {}).get('prefix', 'models/')}",
                    sagemaker_session=self.sagemaker_session,
                    dependencies=[os.path.join(os.path.dirname(script_path), 'requirements_gnn.txt')]
                )
            
            # 訓練データのチャネルを準備
            train_channels = {'train': train_data_path}
            if validation_data_path:
                train_channels['validation'] = validation_data_path
            
            # 訓練を実行
            estimator.fit(train_channels)
            
            logger.info("GNNモデルの訓練が完了しました")
            return estimator
        
        except Exception as e:
            logger.error(f"GNNモデル訓練中にエラーが発生しました: {e}")
            raise
    
    def _get_dgl_container_uri(self) -> str:
        """
        DGLコンテナのURIを取得
        
        Returns:
            ECRコンテナイメージURI
        """
        # DGLコンテナのURI（リージョンに応じて変更）
        # 例: 763104351884.dkr.ecr.ap-northeast-1.amazonaws.com/dgl-training:1.0.0-cpu-py3
        region = self.region
        account_id = '763104351884'  # AWS Deep Learning Containers アカウントID
        
        # DGLコンテナのバージョン（最新のものを確認）
        dgl_version = '1.0.0'
        framework = 'dgl'
        device = 'cpu'  # または 'gpu'
        python_version = 'py3'
        
        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{framework}-training:{dgl_version}-{device}-{python_version}"
        
        logger.info(f"DGLコンテナURI: {image_uri}")
        return image_uri
    
    def deploy_model(self, estimator: Any, endpoint_name: str) -> str:
        """
        モデルをエンドポイントにデプロイ
        
        Args:
            estimator: 訓練済みエスティメーター
            endpoint_name: エンドポイント名
        
        Returns:
            エンドポイント名
        """
        try:
            predictor = estimator.deploy(
                initial_instance_count=1,
                instance_type='ml.t2.medium',
                endpoint_name=endpoint_name
            )
            
            logger.info(f"モデルをエンドポイントにデプロイしました: {endpoint_name}")
            return endpoint_name
        
        except Exception as e:
            logger.error(f"モデルのデプロイ中にエラーが発生しました: {e}")
            raise

