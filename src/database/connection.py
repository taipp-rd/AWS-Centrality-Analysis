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
    
    def get_nodes(self, limit: Optional[int] = None, active_channels_only: bool = True) -> List[Dict[str, Any]]:
        """
        ノードデータを取得
        
        Args:
            limit: 取得件数の上限
            active_channels_only: Trueの場合、アクティブなチャネルを持つノードのみ取得
        
        Returns:
            ノードデータのリスト
        """
        if active_channels_only:
            # アクティブなチャネルを持つノードのみ取得
            query = """
            SELECT DISTINCT na.*
            FROM node_announcement na
            INNER JOIN (
                SELECT DISTINCT advertising_nodeid as node_id
                FROM channel_update
                WHERE rp_disabled = false
                AND chan_id NOT IN (SELECT chan_id FROM closed_channel)
                UNION
                SELECT DISTINCT connecting_nodeid as node_id
                FROM channel_update
                WHERE rp_disabled = false
                AND chan_id NOT IN (SELECT chan_id FROM closed_channel)
            ) active_nodes ON na.node_id = active_nodes.node_id
            """
        else:
            query = "SELECT * FROM node_announcement"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_edges(self, limit: Optional[int] = None, active_only: bool = True, 
                  exclude_closed: bool = True, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        エッジ（チャネル）データを取得
        
        Args:
            limit: 取得件数の上限
            active_only: Trueの場合、有効なチャネルのみ取得（rp_disabled = false）
            exclude_closed: Trueの場合、閉じられたチャネルを除外
            days_back: 指定した日数以内のデータのみ取得（Noneの場合は制限なし）
        
        Returns:
            エッジデータのリスト
        """
        # 基本クエリ：有効なチャネルのみ
        if active_only:
            query = """
            SELECT DISTINCT ON (chan_id) cu.*
            FROM channel_update cu
            WHERE cu.rp_disabled = false
            """
        else:
            query = """
            SELECT DISTINCT ON (chan_id) cu.*
            FROM channel_update cu
            WHERE 1=1
            """
        
        # 閉じられたチャネルを除外
        if exclude_closed:
            query += """
            AND cu.chan_id NOT IN (SELECT chan_id FROM closed_channel)
            """
        
        # 最新の更新のみを取得（同じchan_idで最新のtimestampのもの）
        query += """
        ORDER BY cu.chan_id, cu.timestamp DESC
        """
        
        # 日数制限がある場合
        if days_back:
            query = f"""
            SELECT DISTINCT ON (chan_id) cu.*
            FROM channel_update cu
            WHERE cu.rp_disabled = false
            AND cu.timestamp >= EXTRACT(EPOCH FROM NOW() - INTERVAL '{days_back} days')::bigint
            """
            if exclude_closed:
                query += """
                AND cu.chan_id NOT IN (SELECT chan_id FROM closed_channel)
                """
            query += """
            ORDER BY cu.chan_id, cu.timestamp DESC
            """
        
        if limit:
            query += f" LIMIT {limit}"
        
        return self.execute_query(query)
    
    def get_graph_structure(self, exclude_closed: bool = True) -> Dict[str, Any]:
        """
        グラフ構造の基本情報を取得
        
        Args:
            exclude_closed: Trueの場合、閉じられたチャネルを除外
        
        Returns:
            グラフ構造情報（ノード数、エッジ数など）
        """
        # アクティブなチャネルを持つノード数を取得
        node_count_query = """
        SELECT COUNT(DISTINCT na.node_id) as count
        FROM node_announcement na
        INNER JOIN (
            SELECT DISTINCT advertising_nodeid as node_id
            FROM channel_update
            WHERE rp_disabled = false
        """
        if exclude_closed:
            node_count_query += """
            AND chan_id NOT IN (SELECT chan_id FROM closed_channel)
            """
        node_count_query += """
            UNION
            SELECT DISTINCT connecting_nodeid as node_id
            FROM channel_update
            WHERE rp_disabled = false
        """
        if exclude_closed:
            node_count_query += """
            AND chan_id NOT IN (SELECT chan_id FROM closed_channel)
            """
        node_count_query += """
        ) active_nodes ON na.node_id = active_nodes.node_id
        """
        
        # アクティブなチャネル数（重複を除いた最新の更新のみ）
        edge_count_query = """
        SELECT COUNT(DISTINCT chan_id) as count
        FROM (
            SELECT DISTINCT ON (chan_id) chan_id
            FROM channel_update
            WHERE rp_disabled = false
        """
        if exclude_closed:
            edge_count_query += """
            AND chan_id NOT IN (SELECT chan_id FROM closed_channel)
            """
        edge_count_query += """
            ORDER BY chan_id, timestamp DESC
        ) active_channels
        """
        
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
