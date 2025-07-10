#!/usr/bin/env python3
"""
CLI 音響テストユーティリティ

現行 Geocussion-SP の音響サブシステム (src.sound) を CUI から直接操作して
問題切り分けを行えるようにする小さなインタラクティブツールです。

使い方:
    python sound_cli.py            # 対話モードを起動

操作方法:
    a|b|c|d のアルファベット (楽器選択) + 1|2|3|4 の数字 (音階選択) を入力します。
    例) b2 → 楽器 "b" に対応する Bell 音色 で 音階 #2 を再生

    quit / q / exit を入力すると終了します。

アルファベット – 楽器対応表:
    a: Marimba
    b: Bell
    c: Synth Pad
    d: Drum (Percussion)

数字 – 音階対応表 (MIDI ノート番号):
    1: C4 (60)
    2: D4 (62)
    3: E4 (64)
    4: G4 (67)

このマッピングは暫定的なもので、既存の AudioMapper を経由せず
AudioParameters を直接生成し AudioSynthesizer に渡しています。
今後、詳細な音色マッピングや Velocity 変化を検証する際は
必要に応じて置き換えてください。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# プロジェクトの src/ ディレクトリを import パスに追加
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---- Geocussion-SP サウンドモジュール -------------------------------------------------
try:
    from src.sound import (
        InstrumentType,
        AudioParameters,
        create_audio_synthesizer,
    )
except ImportError as exc:  # pragma: no cover – module path mis-config
    print("[ERROR] src.sound をインポートできませんでした。プロジェクトルートで実行してください。\n", exc)
    sys.exit(1)

# --------------------------------------------------------------------------------------
# 定数 / マッピング
# --------------------------------------------------------------------------------------
INSTRUMENT_MAP: Dict[str, InstrumentType] = {
    "a": InstrumentType.MARIMBA,
    "b": InstrumentType.BELL,
    "c": InstrumentType.SYNTH_PAD,
    "d": InstrumentType.DRUM,
}

NOTE_MAP: Dict[str, int] = {
    "1": 60,  # C4
    "2": 62,  # D4
    "3": 64,  # E4
    "4": 67,  # G4
}

VELOCITY_DEFAULT = 0.8  # 0.0-1.0
DURATION_DEFAULT = 1.2  # seconds

# --------------------------------------------------------------------------------------
# ユーティリティ関数
# --------------------------------------------------------------------------------------

def build_audio_params(inst: InstrumentType, midi_note: int) -> AudioParameters:
    """Instrument と MIDI ノートから AudioParameters を構築"""
    timestamp = time.perf_counter()
    pitch = float(midi_note)

    return AudioParameters(
        pitch=pitch,
        velocity=VELOCITY_DEFAULT,
        duration=DURATION_DEFAULT,
        instrument=inst,
        timbre=0.5,
        brightness=0.5,
        pan=0.0,
        distance=0.0,
        reverb=0.3,
        attack=0.01,
        decay=0.1,
        sustain=0.7,
        release=0.2,
        event_id=f"cli_{int(timestamp * 1e6):d}",
        hand_id="cli",
        timestamp=timestamp,
        gain=1.0,
    )


# --------------------------------------------------------------------------------------
# メイン処理
# --------------------------------------------------------------------------------------

def main() -> None:
    print("==============================================")
    print("🎹 Geocussion-SP CLI Sound Tester")
    print("==============================================")
    print("楽器 (a-d) と 音階 (1-4) を組み合わせて入力してください。例: b2")
    print("q / quit / exit で終了します。\n")

    # --- Audio engine (pygame version) -----------------------------------
    synth = create_audio_synthesizer(sample_rate=44100, buffer_size=512, max_polyphony=32)
    if not synth.start_engine():
        print("[ERROR] Audio engine を起動できませんでした。pygame がインストールされているか確認してください。")
        sys.exit(1)

    try:
        while True:
            # 終了したボイスをクリーンアップ
            try:
                synth.cleanup_finished_voices()
            except Exception:
                pass

            try:
                user_input = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input in {"q", "quit", "exit"}:
                break

            if len(user_input) != 2:
                print("[WARN] 2 文字で入力してください (例: a1, b3)")
                continue

            inst_code, note_code = user_input[0], user_input[1]
            if inst_code not in INSTRUMENT_MAP:
                print(f"[WARN] 無効な楽器コード: {inst_code}. 使用可能: {', '.join(INSTRUMENT_MAP.keys())}")
                continue
            if note_code not in NOTE_MAP:
                print(f"[WARN] 無効な音階コード: {note_code}. 使用可能: {', '.join(NOTE_MAP.keys())}")
                continue

            instrument = INSTRUMENT_MAP[inst_code]
            midi_note = NOTE_MAP[note_code]
            params = build_audio_params(instrument, midi_note)

            voice_id = synth.play_note(params)

            if voice_id:
                print(
                    f"[OK] 再生開始: {instrument.value} (MIDI {midi_note}) -> VoiceID {voice_id}"
                )
            else:
                print("[ERROR] 再生に失敗しました。")

    finally:
        # 終了処理
        try:
            synth.stop_engine()
        except Exception:
            pass
        print("バイバイ 👋")


if __name__ == "__main__":
    main() 