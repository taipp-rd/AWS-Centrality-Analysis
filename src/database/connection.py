"""
RDS PostgreSQLデータベース接続モジュール
"""

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import yaml
import os
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """RDS PostgreSQLデータベースへの接続を管理するクラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初期化
        
        Args:
            config_path: 設定ファイルのパス（Noneの場合はデフォルトパスを使用）
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config',
                'config.yaml'
            )
        
        self.config = self._load_config(config_path)
        self.connection_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._initialize_pool()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """設定ファイルを読み込む"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('database', {})
        except FileNotFoundError:
            logger.error(f"設定ファイルが見つかりません: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"設定ファイルの読み込みエラー: {e}")
            raise
    
    def _initialize_pool(self, min_conn: int = 1, max_conn: int = 5):
        """接続プールを初期化"""
        try:
            self.connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['dbname'],
                user=self.config['user'],
                password=self.config['password'],
                sslmode=self.config.get('sslmode', 'prefer'),
                connect_timeout=self.config.get('connect_timeout', 10)
            )
            logger.info("データベース接続プールを初期化しました")
        except Exception as e:
            logger.error(f"データベース接続プールの初期化に失敗しました: {e}")
            raise
    
    def get_connection(self):
        """接続プールから接続を取得"""
        if self.connection_pool is None:
            self._initialize_pool()
        return self.connection_pool.getconn()
    
    def return_connection(self, conn):
        """接続をプールに返す"""
        if self.connection_pool:
            self.connection_pool.putconn(conn)
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        クエリを実行して結果を返す
        
        Args:
            query: SQLクエリ
            params: クエリパラメータ
        
        Returns:
            クエリ結果のリスト
        """
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"クエリ実行エラー: {e}")
            logger.error(f"クエリ: {query}")
            raise
        finally:
            if conn:
                self.return_connection(conn)
    
    def execute_query_single(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """
        単一の結果を返すクエリを実行
        
        Args:
            query: SQLクエリ
            params: クエリパラメータ
        
        Returns:
            クエリ結果（単一の辞書）またはNone
        """
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    def get_nodes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        ノードデータを取得
        
        Args:
            limit: 取得件数の上限
        
        Returns:
            ノードデータのリスト
        """
        query = "SELECT * FROM nodes"
        if limit:
            query += f" LIMIT {limit}"
        return self.execute_query(query)
    
    def get_edges(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        エッジ（チャネル）データを取得
        
        Args:
            limit: 取得件数の上限
        
        Returns:
            エッジデータのリスト
        """
        query = "SELECT * FROM edges"
        if limit:
            query += f" LIMIT {limit}"
        return self.execute_query(query)
    
    def get_graph_structure(self) -> Dict[str, Any]:
        """
        グラフ構造の基本情報を取得
        
        Returns:
            グラフ構造情報（ノード数、エッジ数など）
        """
        node_count_query = "SELECT COUNT(*) as count FROM nodes"
        edge_count_query = "SELECT COUNT(*) as count FROM edges"
        
        node_count = self.execute_query_single(node_count_query)
        edge_count = self.execute_query_single(edge_count_query)
        
        return {
            'node_count': node_count['count'] if node_count else 0,
            'edge_count': edge_count['count'] if edge_count else 0
        }
    
    def close(self):
        """接続プールを閉じる"""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("データベース接続プールを閉じました")

