#!/usr/bin/env python3
"""
手検出重複排除フィルタ

同一の物理的な手が複数の手として検出されることを防ぎ、
「一打＝一音」の実現を支援する重複排除システムを提供します。
"""

import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

from .hands2d import HandDetectionResult
from ..data_types import HandednessType
from .. import get_logger

logger = get_logger(__name__)


@dataclass
class HandCluster:
    """手クラスター（重複手の統合表現）"""
    representative_hand: HandDetectionResult  # 代表的な手
    cluster_hands: List[HandDetectionResult]  # クラスター内の手のリスト
    confidence_score: float  # 統合信頼度
    cluster_center: np.ndarray  # クラスター中心座標
    cluster_size: float  # クラスターサイズ
    handedness: HandednessType  # 手の種類
    last_updated: float  # 最終更新時刻


class HandDeduplicationFilter:
    """手検出重複排除フィルタ"""
    
    def __init__(
        self,
        distance_threshold: float = 0.15,  # 15cm以内は同じ手と判定
        confidence_boost: float = 0.1,     # 統合による信頼度向上
        max_cluster_age: float = 1.0,      # 1秒でクラスター無効化
        handedness_weight: float = 2.0,    # 左右一致時の重み
        size_similarity_threshold: float = 0.3  # サイズ類似度閾値
    ):
        """
        初期化
        
        Args:
            distance_threshold: 同一手判定距離閾値
            confidence_boost: 統合時の信頼度向上値
            max_cluster_age: クラスター最大有効時間
            handedness_weight: 左右一致重み
            size_similarity_threshold: サイズ類似度閾値
        """
        self.distance_threshold = distance_threshold
        self.confidence_boost = confidence_boost
        self.max_cluster_age = max_cluster_age
        self.handedness_weight = handedness_weight
        self.size_similarity_threshold = size_similarity_threshold
        
        # 統計情報
        self.stats = {
            'total_hands_processed': 0,
            'duplicates_removed': 0,
            'clusters_created': 0,
            'clusters_merged': 0,
            'confidence_improvements': 0
        }
        
        logger.info(f"HandDeduplicationFilter initialized with {distance_threshold}m threshold")
    
    def filter_duplicate_hands(self, hands: List[HandDetectionResult]) -> List[HandDetectionResult]:
        """
        重複手を排除してフィルタリング
        
        Args:
            hands: 検出された手のリスト
            
        Returns:
            重複排除後の手のリスト
        """
        if len(hands) <= 1:
            self.stats['total_hands_processed'] += len(hands)
            return hands
        
        start_time = time.perf_counter()
        current_time = time.time()
        
        try:
            # クラスタリング実行
            clusters = self._create_hand_clusters(hands)
            
            # 各クラスターから代表手を選択
            filtered_hands = []
            duplicates_removed = 0
            
            for cluster in clusters:
                representative = self._select_representative_hand(cluster)
                filtered_hands.append(representative)
                
                # 重複統計更新
                cluster_size = len(cluster.cluster_hands)
                if cluster_size > 1:
                    duplicates_removed += cluster_size - 1
                    self.stats['duplicates_removed'] += duplicates_removed
                    logger.debug(f"Merged {cluster_size} hands into 1 representative")
            
            # 統計更新
            self.stats['total_hands_processed'] += len(hands)
            self.stats['clusters_created'] += len(clusters)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Hand deduplication: {len(hands)} -> {len(filtered_hands)} hands ({processing_time:.2f}ms)")
            
            return filtered_hands
            
        except Exception as e:
            logger.error(f"Hand deduplication error: {e}")
            return hands  # エラー時は元の検出結果を返す
    
    def _create_hand_clusters(self, hands: List[HandDetectionResult]) -> List[HandCluster]:
        """手のクラスタリングを実行"""
        clusters = []
        used_hands = set()
        
        for i, hand in enumerate(hands):
            if i in used_hands:
                continue
            
            # 新しいクラスターを開始
            cluster_hands = [hand]
            used_hands.add(i)
            
            # 近傍の手を探してクラスターに追加
            for j, other_hand in enumerate(hands):
                if j in used_hands or i == j:
                    continue
                
                if self._should_merge_hands(hand, other_hand):
                    cluster_hands.append(other_hand)
                    used_hands.add(j)
            
            # クラスター作成
            if cluster_hands:
                cluster = self._build_cluster(cluster_hands)
                clusters.append(cluster)
        
        return clusters
    
    def _should_merge_hands(self, hand1: HandDetectionResult, hand2: HandDetectionResult) -> bool:
        """2つの手をマージすべきかどうかを判定"""
        
        # 1. 距離チェック
        center1 = self._get_hand_center(hand1)
        center2 = self._get_hand_center(hand2)
        distance = np.linalg.norm(center1 - center2)
        
        if distance > self.distance_threshold:
            return False
        
        # 2. 左右一致チェック
        handedness_match = hand1.handedness == hand2.handedness
        if not handedness_match:
            # 左右が異なる場合は距離閾値を厳しくする
            if distance > self.distance_threshold * 0.7:
                return False
        
        # 3. サイズ類似度チェック
        size1 = self._get_hand_size(hand1)
        size2 = self._get_hand_size(hand2)
        size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0.0
        
        if size_ratio < self.size_similarity_threshold:
            return False
        
        # 4. 信頼度による重み付け判定
        confidence_factor = (hand1.confidence + hand2.confidence) / 2.0
        adjusted_threshold = self.distance_threshold * (2.0 - confidence_factor)  # 信頼度が高いほど厳しく
        
        return distance <= adjusted_threshold
    
    def _build_cluster(self, cluster_hands: List[HandDetectionResult]) -> HandCluster:
        """手のクラスターを構築"""
        
        # 代表手選択（最高信頼度）
        representative = max(cluster_hands, key=lambda h: h.confidence)
        
        # クラスター中心計算
        centers = [self._get_hand_center(hand) for hand in cluster_hands]
        cluster_center = np.mean(centers, axis=0)
        
        # クラスターサイズ計算
        distances = [np.linalg.norm(center - cluster_center) for center in centers]
        cluster_size = np.max(distances) if distances else 0.0
        
        # 統合信頼度計算
        base_confidence = representative.confidence
        cluster_count_boost = min(len(cluster_hands) - 1, 3) * self.confidence_boost  # 最大3倍まで
        confidence_score = min(1.0, base_confidence + cluster_count_boost)
        
        if confidence_score > base_confidence:
            self.stats['confidence_improvements'] += 1
        
        return HandCluster(
            representative_hand=representative,
            cluster_hands=cluster_hands,
            confidence_score=confidence_score,
            cluster_center=cluster_center,
            cluster_size=cluster_size,
            handedness=representative.handedness,
            last_updated=time.time()
        )
    
    def _select_representative_hand(self, cluster: HandCluster) -> HandDetectionResult:
        """クラスターから代表手を選択"""
        representative = cluster.representative_hand
        
        # 統合後の信頼度で更新
        updated_hand = HandDetectionResult(
            id=representative.id,  # IDは代表手のものを維持
            landmarks=representative.landmarks,
            handedness=representative.handedness,
            confidence=cluster.confidence_score,  # 統合信頼度に更新
            bounding_box=representative.bounding_box,
            timestamp_ms=representative.timestamp_ms,
            is_tracked=representative.is_tracked
        )
        
        return updated_hand
    
    def _get_hand_center(self, hand: HandDetectionResult) -> np.ndarray:
        """手の中心座標を取得（正規化座標）"""
        if hand.landmarks and len(hand.landmarks) > 0:
            # ランドマークの重心を計算（正規化座標 0-1）
            x_coords = [lm.x for lm in hand.landmarks]
            y_coords = [lm.y for lm in hand.landmarks]
            return np.array([np.mean(x_coords), np.mean(y_coords)])
        else:
            # バウンディングボックスの中心を正規化座標に変換
            x, y, w, h = hand.bounding_box
            # 一般的な画像サイズを仮定してピクセル座標を正規化
            image_width, image_height = 640, 480  # デフォルト値
            center_x = (x + w/2) / image_width
            center_y = (y + h/2) / image_height
            return np.array([center_x, center_y])
    
    def _get_hand_size(self, hand: HandDetectionResult) -> float:
        """手のサイズを推定（正規化座標）"""
        x, y, w, h = hand.bounding_box
        # 一般的な画像サイズで正規化
        image_width, image_height = 640, 480
        norm_w = w / image_width
        norm_h = h / image_height
        return np.sqrt(norm_w * norm_w + norm_h * norm_h)  # 正規化対角線長
    
    def get_stats(self) -> Dict[str, int]:
        """統計情報を取得"""
        return self.stats.copy()
    
    def reset_stats(self):
        """統計情報をリセット"""
        for key in self.stats:
            self.stats[key] = 0


# 便利関数
def filter_duplicate_hands(
    hands: List[HandDetectionResult],
    distance_threshold: float = 0.15,
    confidence_boost: float = 0.1
) -> List[HandDetectionResult]:
    """
    手検出重複排除の簡易インターフェース
    
    Args:
        hands: 検出された手のリスト
        distance_threshold: 同一手判定距離閾値
        confidence_boost: 統合時の信頼度向上値
        
    Returns:
        重複排除後の手のリスト
    """
    filter_instance = HandDeduplicationFilter(
        distance_threshold=distance_threshold,
        confidence_boost=confidence_boost
    )
    return filter_instance.filter_duplicate_hands(hands)


def create_hand_deduplication_filter(
    distance_threshold: float = 0.15,
    confidence_boost: float = 0.1,
    max_cluster_age: float = 1.0
) -> HandDeduplicationFilter:
    """
    手検出重複排除フィルタを作成
    
    Args:
        distance_threshold: 同一手判定距離閾値
        confidence_boost: 統合時の信頼度向上値
        max_cluster_age: クラスター最大有効時間
        
    Returns:
        HandDeduplicationFilter インスタンス
    """
    return HandDeduplicationFilter(
        distance_threshold=distance_threshold,
        confidence_boost=confidence_boost,
        max_cluster_age=max_cluster_age
    ) 