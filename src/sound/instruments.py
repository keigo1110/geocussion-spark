#!/usr/bin/env python3
"""
全楽器対応マリンバプログラム（pygame版）
様々な楽器の音色をシミュレート
"""

import numpy as np
import pygame
import time
from enum import Enum

# 音階の周波数
NOTES = {
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
    'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
}


class InstrumentType(Enum):
    """利用可能な楽器タイプ"""
    # 打楽器系
    MARIMBA = "marimba"          # マリンバ（木琴）
    VIBRAPHONE = "vibraphone"    # ビブラフォン（鉄琴）
    GLOCKENSPIEL = "glockenspiel"  # グロッケンシュピール
    XYLOPHONE = "xylophone"      # シロフォン
    
    # シンセサイザー系
    SYNTH_PAD = "synth_pad"      # シンセパッド
    SYNTH_LEAD = "synth_lead"    # シンセリード
    SYNTH_BASS = "synth_bass"    # シンセベース
    
    # ベル・金属系
    BELL = "bell"                # ベル
    CHURCH_BELL = "church_bell"  # 教会の鐘
    CRYSTAL = "crystal"          # クリスタル
    MUSIC_BOX = "music_box"      # オルゴール
    
    # 自然音系
    WATER_DROP = "water_drop"    # 水滴
    WIND = "wind"                # 風
    RAIN = "rain"                # 雨
    
    # 弦楽器系
    STRING = "string"            # 弦楽器
    HARP = "harp"                # ハープ
    GUITAR = "guitar"            # ギター
    
    # 管楽器系
    FLUTE = "flute"              # フルート
    PAN_FLUTE = "pan_flute"      # パンフルート
    
    # ドラム・パーカッション
    DRUM = "drum"                # ドラム
    TIMPANI = "timpani"          # ティンパニ
    CONGA = "conga"              # コンガ
    
    # ピアノ系
    PIANO = "piano"              # ピアノ
    ELECTRIC_PIANO = "electric_piano"  # エレピ


class InstrumentSynthesizer:
    """楽器シンセサイザー"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def generate_wave(self, instrument: InstrumentType, frequency: float, 
                     duration: float = 2.0, velocity: float = 0.7) -> np.ndarray:
        """指定された楽器の波形を生成"""
        
        # 時間軸
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # 楽器ごとの波形生成
        if instrument == InstrumentType.MARIMBA:
            wave = self._marimba(t, frequency)
        elif instrument == InstrumentType.VIBRAPHONE:
            wave = self._vibraphone(t, frequency)
        elif instrument == InstrumentType.GLOCKENSPIEL:
            wave = self._glockenspiel(t, frequency)
        elif instrument == InstrumentType.XYLOPHONE:
            wave = self._xylophone(t, frequency)
        elif instrument == InstrumentType.SYNTH_PAD:
            wave = self._synth_pad(t, frequency)
        elif instrument == InstrumentType.SYNTH_LEAD:
            wave = self._synth_lead(t, frequency)
        elif instrument == InstrumentType.SYNTH_BASS:
            wave = self._synth_bass(t, frequency)
        elif instrument == InstrumentType.BELL:
            wave = self._bell(t, frequency)
        elif instrument == InstrumentType.CHURCH_BELL:
            wave = self._church_bell(t, frequency)
        elif instrument == InstrumentType.CRYSTAL:
            wave = self._crystal(t, frequency)
        elif instrument == InstrumentType.MUSIC_BOX:
            wave = self._music_box(t, frequency)
        elif instrument == InstrumentType.WATER_DROP:
            wave = self._water_drop(t, frequency)
        elif instrument == InstrumentType.WIND:
            wave = self._wind(t, frequency)
        elif instrument == InstrumentType.RAIN:
            wave = self._rain(t, frequency)
        elif instrument == InstrumentType.STRING:
            wave = self._string(t, frequency)
        elif instrument == InstrumentType.HARP:
            wave = self._harp(t, frequency)
        elif instrument == InstrumentType.GUITAR:
            wave = self._guitar(t, frequency)
        elif instrument == InstrumentType.FLUTE:
            wave = self._flute(t, frequency)
        elif instrument == InstrumentType.PAN_FLUTE:
            wave = self._pan_flute(t, frequency)
        elif instrument == InstrumentType.DRUM:
            wave = self._drum(t, frequency)
        elif instrument == InstrumentType.TIMPANI:
            wave = self._timpani(t, frequency)
        elif instrument == InstrumentType.CONGA:
            wave = self._conga(t, frequency)
        elif instrument == InstrumentType.PIANO:
            wave = self._piano(t, frequency)
        elif instrument == InstrumentType.ELECTRIC_PIANO:
            wave = self._electric_piano(t, frequency)
        else:
            # デフォルト
            wave = np.sin(2 * np.pi * frequency * t)
        
        # 音量調整
        wave = wave * velocity
        
        # クリッピング防止
        wave = np.clip(wave, -1.0, 1.0)
        
        # 16ビット整数に変換
        wave = (wave * 32767 * 0.8).astype(np.int16)
        
        # ステレオ化
        stereo = np.zeros((len(wave), 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return stereo
    
    def _marimba(self, t, freq):
        """マリンバ：木製の温かい音"""
        fundamental = np.sin(2 * np.pi * freq * t)
        h2 = 0.4 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.15 * np.sin(2 * np.pi * freq * 3 * t)
        wave = fundamental + h2 + h3
        
        # エンベロープ（木琴特有の減衰）
        env = np.exp(-3 * t) * (1 - np.exp(-500 * t))
        return wave * env
    
    def _vibraphone(self, t, freq):
        """ビブラフォン：金属製でビブラート付き"""
        fundamental = np.sin(2 * np.pi * freq * t)
        h2 = 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        
        # ビブラート
        vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * t)
        wave = (fundamental + h2 + h3) * vibrato
        
        # 長めの減衰
        env = np.exp(-1.5 * t) * (1 - np.exp(-200 * t))
        return wave * env
    
    def _glockenspiel(self, t, freq):
        """グロッケンシュピール：明るくキラキラした音"""
        fundamental = np.sin(2 * np.pi * freq * t)
        h2 = 0.6 * np.sin(2 * np.pi * freq * 2.1 * t)  # 少しデチューン
        h3 = 0.3 * np.sin(2 * np.pi * freq * 3.05 * t)
        h4 = 0.2 * np.sin(2 * np.pi * freq * 4.2 * t)
        wave = fundamental + h2 + h3 + h4
        
        # 短い減衰
        env = np.exp(-4 * t) * (1 - np.exp(-1000 * t))
        return wave * env
    
    def _xylophone(self, t, freq):
        """シロフォン：硬い木の音"""
        fundamental = np.sin(2 * np.pi * freq * t)
        h2 = 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.2 * np.sin(2 * np.pi * freq * 3 * t)
        
        # クリック音を追加
        click = 0.1 * np.random.normal(0, 1, len(t)) * np.exp(-100 * t)
        wave = fundamental + h2 + h3 + click
        
        env = np.exp(-5 * t) * (1 - np.exp(-800 * t))
        return wave * env
    
    def _synth_pad(self, t, freq):
        """シンセパッド：厚みのある持続音"""
        # デチューンした複数のオシレーター
        osc1 = np.sin(2 * np.pi * freq * t)
        osc2 = np.sin(2 * np.pi * freq * 1.01 * t)
        osc3 = np.sin(2 * np.pi * freq * 0.99 * t)
        
        # ローパスフィルター風（簡易版）
        wave = (osc1 + 0.7 * osc2 + 0.7 * osc3) / 2.4
        
        # ゆっくりしたエンベロープ
        env = (1 - np.exp(-2 * t)) * np.exp(-0.5 * t)
        return wave * env
    
    def _synth_lead(self, t, freq):
        """シンセリード：鋭い音"""
        # ノコギリ波（簡易版）
        wave = 2 * (t * freq % 1) - 1
        
        # 高調波を追加
        wave += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        
        env = (1 - np.exp(-50 * t)) * np.exp(-1 * t)
        return wave * env * 0.5
    
    def _synth_bass(self, t, freq):
        """シンセベース：低音"""
        # サブオシレーター付き
        sub = np.sin(2 * np.pi * freq * 0.5 * t)
        fundamental = np.sin(2 * np.pi * freq * t)
        h2 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        
        wave = 0.5 * sub + fundamental + h2
        
        env = (1 - np.exp(-20 * t)) * np.exp(-2 * t)
        return wave * env * 0.7
    
    def _bell(self, t, freq):
        """ベル：FM合成風"""
        mod_freq = freq * 3.5
        mod_amp = freq * 0.5 * np.exp(-2 * t)
        modulator = mod_amp * np.sin(2 * np.pi * mod_freq * t)
        carrier = np.sin(2 * np.pi * (freq + modulator) * t)
        
        # 倍音
        h2 = 0.3 * np.sin(2 * np.pi * freq * 2.1 * t)
        wave = carrier + h2
        
        env = np.exp(-1.5 * t) * (1 - np.exp(-100 * t))
        return wave * env
    
    def _church_bell(self, t, freq):
        """教会の鐘：重厚な音"""
        # 複数の非調和倍音
        f1 = np.sin(2 * np.pi * freq * t)
        f2 = 0.5 * np.sin(2 * np.pi * freq * 2.4 * t)
        f3 = 0.3 * np.sin(2 * np.pi * freq * 3.7 * t)
        f4 = 0.2 * np.sin(2 * np.pi * freq * 5.1 * t)
        
        wave = f1 + f2 + f3 + f4
        
        # ゆっくりした減衰
        env = np.exp(-0.5 * t) * (1 - np.exp(-50 * t))
        return wave * env
    
    def _crystal(self, t, freq):
        """クリスタル：澄んだ音"""
        # 純正律の倍音
        h1 = np.sin(2 * np.pi * freq * t)
        h2 = 0.7 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.5 * np.sin(2 * np.pi * freq * 3 * t)
        h4 = 0.3 * np.sin(2 * np.pi * freq * 4 * t)
        h5 = 0.2 * np.sin(2 * np.pi * freq * 5 * t)
        
        wave = h1 + h2 + h3 + h4 + h5
        
        # キラキラ感
        shimmer = 1 + 0.05 * np.sin(2 * np.pi * 7 * t)
        wave *= shimmer
        
        env = np.exp(-2 * t) * (1 - np.exp(-300 * t))
        return wave * env * 0.5
    
    def _music_box(self, t, freq):
        """オルゴール：懐かしい音"""
        # 基音と5度上
        fundamental = np.sin(2 * np.pi * freq * t)
        fifth = 0.4 * np.sin(2 * np.pi * freq * 1.5 * t)
        octave = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        
        wave = fundamental + fifth + octave
        
        # 機械的な減衰
        env = np.exp(-3 * t) * (1 - np.exp(-500 * t))
        
        # わずかな揺らぎ
        flutter = 1 + 0.01 * np.sin(2 * np.pi * 3 * t)
        return wave * env * flutter
    
    def _water_drop(self, t, freq):
        """水滴：ピッチベンド"""
        # 周波数が下がる
        bent_freq = freq * (1 + 0.5 * np.exp(-10 * t))
        wave = np.sin(2 * np.pi * bent_freq * t)
        
        # 水の響き
        resonance = 0.3 * np.sin(2 * np.pi * bent_freq * 2.1 * t)
        wave += resonance
        
        env = np.exp(-5 * t) * (1 - np.exp(-200 * t))
        return wave * env
    
    def _wind(self, t, freq):
        """風：フィルターノイズ"""
        # ホワイトノイズ
        noise = np.random.normal(0, 1, len(t))
        
        # 簡易バンドパスフィルター（周波数に依存）
        cutoff = 0.1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        filtered = noise * cutoff
        
        # トーン成分を少し追加
        tone = 0.1 * np.sin(2 * np.pi * freq * t)
        
        wave = filtered + tone
        env = (1 - np.exp(-1 * t)) * np.exp(-0.5 * t)
        return wave * env * 0.3
    
    def _rain(self, t, freq):
        """雨：複数の水滴"""
        wave = np.zeros_like(t)
        
        # ランダムな水滴
        for i in range(5):
            delay = i * 0.1
            if delay < t[-1]:
                drop_t = t - delay
                drop_t[drop_t < 0] = 0
                drop_freq = freq * (1 + i * 0.1)
                drop = np.sin(2 * np.pi * drop_freq * drop_t) * np.exp(-10 * drop_t)
                wave += drop * 0.2
        
        return wave
    
    def _string(self, t, freq):
        """弦楽器：豊かな倍音"""
        # 複数の倍音
        harmonics = []
        for i in range(1, 8):
            amp = 1.0 / i
            harmonics.append(amp * np.sin(2 * np.pi * freq * i * t))
        
        wave = sum(harmonics)
        
        # ボウイング（弓）効果
        env = (1 - np.exp(-5 * t)) * np.exp(-0.5 * t)
        return wave * env * 0.3
    
    def _harp(self, t, freq):
        """ハープ：明るい弦の音"""
        # 倍音構成
        h1 = np.sin(2 * np.pi * freq * t)
        h2 = 0.5 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.25 * np.sin(2 * np.pi * freq * 3 * t)
        h4 = 0.125 * np.sin(2 * np.pi * freq * 4 * t)
        
        wave = h1 + h2 + h3 + h4
        
        # プラック（はじく）効果
        pluck_env = np.exp(-4 * t) * (1 - np.exp(-100 * t))
        return wave * pluck_env
    
    def _guitar(self, t, freq):
        """ギター：温かい弦の音"""
        # カルプラス・ストロング風
        h1 = np.sin(2 * np.pi * freq * t)
        h2 = 0.6 * np.sin(2 * np.pi * freq * 2 * t)
        h3 = 0.3 * np.sin(2 * np.pi * freq * 3 * t)
        
        # わずかなデチューン
        detune = 1 + 0.002 * np.sin(2 * np.pi * 1.5 * t)
        wave = (h1 + h2 + h3) * detune
        
        # プラックエンベロープ
        env = np.exp(-2 * t) * (1 - np.exp(-50 * t))
        return wave * env
    
    def _flute(self, t, freq):
        """フルート：純粋な音"""
        # 基音中心、少ない倍音
        fundamental = np.sin(2 * np.pi * freq * t)
        h2 = 0.1 * np.sin(2 * np.pi * freq * 2 * t)
        
        # ブレス効果（ノイズ）
        breath = 0.02 * np.random.normal(0, 1, len(t))
        
        wave = fundamental + h2 + breath
        
        # ソフトなアタック
        env = (1 - np.exp(-10 * t)) * np.exp(-0.5 * t)
        return wave * env
    
    def _pan_flute(self, t, freq):
        """パンフルート：息の音が混じる"""
        # 基音
        fundamental = np.sin(2 * np.pi * freq * t)
        
        # 息のノイズ（多め）
        breath = 0.1 * np.random.normal(0, 1, len(t))
        breath_filtered = breath * np.exp(-5 * t)
        
        wave = fundamental + breath_filtered
        
        env = (1 - np.exp(-5 * t)) * np.exp(-0.7 * t)
        return wave * env
    
    def _drum(self, t, freq):
        """ドラム：打撃音"""
        # トーン成分（低音）
        tone = np.sin(2 * np.pi * 60 * t)
        
        # ノイズ成分
        noise = np.random.normal(0, 1, len(t))
        
        # ピッチベンド
        pitch_env = np.exp(-20 * t)
        bent_tone = np.sin(2 * np.pi * 60 * (1 + 2 * pitch_env) * t)
        
        wave = 0.5 * bent_tone + 0.5 * noise
        
        # パンチのあるエンベロープ
        env = np.exp(-15 * t)
        return wave * env
    
    def _timpani(self, t, freq):
        """ティンパニ：調律可能な太鼓"""
        # 基音（調律された周波数）
        fundamental = np.sin(2 * np.pi * freq * t)
        
        # 倍音
        h2 = 0.3 * np.sin(2 * np.pi * freq * 1.5 * t)
        h3 = 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        
        # 打撃ノイズ
        impact = 0.2 * np.random.normal(0, 1, len(t)) * np.exp(-50 * t)
        
        wave = fundamental + h2 + h3 + impact
        
        # 長めの減衰
        env = np.exp(-1 * t) * (1 - np.exp(-100 * t))
        return wave * env
    
    def _conga(self, t, freq):
        """コンガ：手で叩く太鼓"""
        # 高めの基音
        tone = np.sin(2 * np.pi * freq * 1.5 * t)
        
        # スラップ音
        slap = 0.3 * np.sin(2 * np.pi * freq * 3 * t)
        
        # 皮の振動
        vibration = 0.2 * np.sin(2 * np.pi * freq * 0.8 * t)
        
        wave = tone + slap + vibration
        
        # 短い減衰
        env = np.exp(-8 * t) * (1 - np.exp(-200 * t))
        return wave * env
    
    def _piano(self, t, freq):
        """ピアノ：豊かな倍音と共鳴"""
        # リアルな倍音構成
        harmonics = []
        harmonic_amps = [1.0, 0.4, 0.2, 0.1, 0.08, 0.06, 0.04]
        
        for i, amp in enumerate(harmonic_amps, 1):
            # わずかなストレッチチューニング
            stretched_freq = freq * i * (1 + 0.0002 * i)
            harmonics.append(amp * np.sin(2 * np.pi * stretched_freq * t))
        
        wave = sum(harmonics)
        
        # ハンマーの打撃感
        attack = 1 - np.exp(-200 * t)
        decay = np.exp(-0.5 * t)  # 長い減衰
        env = attack * decay
        
        return wave * env * 0.5
    
    def _electric_piano(self, t, freq):
        """エレクトリックピアノ：ベル的な音"""
        # FM合成風
        mod_freq = freq * 14
        mod_amp = 0.5 * np.exp(-3 * t)
        modulator = mod_amp * np.sin(2 * np.pi * mod_freq * t)
        carrier = np.sin(2 * np.pi * (freq + modulator) * t)
        
        # 特徴的な倍音
        h2 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        
        wave = carrier + h2
        
        # エレピ特有のエンベロープ
        env = (1 - np.exp(-50 * t)) * np.exp(-2 * t)
        return wave * env


def demo_all_instruments():
    """全楽器のデモ演奏"""
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
    synth = InstrumentSynthesizer()
    
    print("=== 全楽器デモンストレーション ===\n")
    
    # カテゴリごとにデモ
    categories = {
        "打楽器系": [
            InstrumentType.MARIMBA,
            InstrumentType.VIBRAPHONE,
            InstrumentType.GLOCKENSPIEL,
            InstrumentType.XYLOPHONE
        ],
        "シンセサイザー系": [
            InstrumentType.SYNTH_PAD,
            InstrumentType.SYNTH_LEAD,
            InstrumentType.SYNTH_BASS
        ],
        "ベル・金属系": [
            InstrumentType.BELL,
            InstrumentType.CHURCH_BELL,
            InstrumentType.CRYSTAL,
            InstrumentType.MUSIC_BOX
        ],
        "自然音系": [
            InstrumentType.WATER_DROP,
            InstrumentType.WIND,
            InstrumentType.RAIN
        ],
        "弦楽器系": [
            InstrumentType.STRING,
            InstrumentType.HARP,
            InstrumentType.GUITAR
        ],
        "管楽器系": [
            InstrumentType.FLUTE,
            InstrumentType.PAN_FLUTE
        ],
        "ドラム系": [
            InstrumentType.DRUM,
            InstrumentType.TIMPANI,
            InstrumentType.CONGA
        ],
        "ピアノ系": [
            InstrumentType.PIANO,
            InstrumentType.ELECTRIC_PIANO
        ]
    }
    
    # 各カテゴリをデモ
    for category, instruments in categories.items():
        print(f"\n【{category}】")
        for instrument in instruments:
            print(f"  {instrument.value}...", end='', flush=True)
            
            # C4の音を生成
            wave = synth.generate_wave(instrument, 261.63, duration=1.5)
            sound = pygame.sndarray.make_sound(wave)
            sound.play()
            time.sleep(1.0)
            print(" ✓")
    
    print("\n=== 楽器による簡単な演奏 ===")
    
    # 異なる楽器で同じメロディを演奏
    melody_notes = [261.63, 293.66, 329.63, 349.23, 392.00]  # C D E F G
    selected_instruments = [
        InstrumentType.PIANO,
        InstrumentType.MARIMBA,
        InstrumentType.GUITAR,
        InstrumentType.FLUTE,
        InstrumentType.BELL
    ]
    
    for instrument in selected_instruments:
        print(f"\n{instrument.value}で演奏:")
        for freq in melody_notes:
            wave = synth.generate_wave(instrument, freq, duration=0.5, velocity=0.6)
            sound = pygame.sndarray.make_sound(wave)
            sound.play()
            time.sleep(0.3)
        time.sleep(0.5)
    
    pygame.mixer.quit()
    print("\nデモ終了！")


def list_all_instruments():
    """利用可能な楽器をリスト表示"""
    print("=== 利用可能な楽器一覧 ===\n")
    
    # カテゴリごとに整理
    categories = {
        "打楽器系 (Percussion)": [
            ("MARIMBA", "マリンバ - 木製の温かい音色"),
            ("VIBRAPHONE", "ビブラフォン - 金属製でビブラート効果"),
            ("GLOCKENSPIEL", "グロッケンシュピール - 明るくキラキラした音"),
            ("XYLOPHONE", "シロフォン - 硬く乾いた木の音")
        ],
        "シンセサイザー系 (Synthesizer)": [
            ("SYNTH_PAD", "シンセパッド - 厚みのある持続音"),
            ("SYNTH_LEAD", "シンセリード - 鋭くカットスルーする音"),
            ("SYNTH_BASS", "シンセベース - 重低音")
        ],
        "ベル・金属系 (Bell/Metal)": [
            ("BELL", "ベル - FM合成的な鐘の音"),
            ("CHURCH_BELL", "教会の鐘 - 重厚で荘厳な音"),
            ("CRYSTAL", "クリスタル - 澄んだ透明感のある音"),
            ("MUSIC_BOX", "オルゴール - 懐かしく機械的な音")
        ],
        "自然音系 (Natural)": [
            ("WATER_DROP", "水滴 - ピッチが下がる水の音"),
            ("WIND", "風 - フィルターされたノイズ"),
            ("RAIN", "雨 - 複数の水滴が重なる音")
        ],
        "弦楽器系 (String)": [
            ("STRING", "弦楽器 - 豊かな倍音を持つ弦"),
            ("HARP", "ハープ - 明るく華やかな弦"),
            ("GUITAR", "ギター - 温かみのある撥弦楽器")
        ],
        "管楽器系 (Wind)": [
            ("FLUTE", "フルート - 純粋で澄んだ音"),
            ("PAN_FLUTE", "パンフルート - 息の音が混じる笛")
        ],
        "ドラム・パーカッション (Drum)": [
            ("DRUM", "ドラム - パンチのある打撃音"),
            ("TIMPANI", "ティンパニ - 調律可能な大太鼓"),
            ("CONGA", "コンガ - 手で叩く中高音の太鼓")
        ],
        "ピアノ系 (Piano)": [
            ("PIANO", "ピアノ - リアルな倍音構成"),
            ("ELECTRIC_PIANO", "エレクトリックピアノ - ベル的な電子音")
        ]
    }
    
    for category, instruments in categories.items():
        print(f"【{category}】")
        for code, description in instruments:
            print(f"  {code:<20} : {description}")
        print()
    
    print("使用例:")
    print("  instrument = InstrumentType.MARIMBA")
    print("  wave = synth.generate_wave(instrument, frequency=440.0)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--list':
            # 楽器一覧を表示
            list_all_instruments()
        elif sys.argv[1] == '--demo':
            # 全楽器のデモ
            demo_all_instruments()
    else:
        # デフォルト：簡単なデモ
        print("使い方:")
        print("  python instruments.py --list  # 楽器一覧")
        print("  python instruments.py --demo  # 全楽器デモ")
        print("\n簡易デモを実行します...\n")
        
        pygame.mixer.init()
        synth = InstrumentSynthesizer()
        
        # いくつかの楽器でC4を演奏
        test_instruments = [
            InstrumentType.PIANO,
            InstrumentType.MARIMBA,
            InstrumentType.BELL,
            InstrumentType.GUITAR,
            InstrumentType.FLUTE
        ]
        
        for inst in test_instruments:
            print(f"♪ {inst.value}")
            wave = synth.generate_wave(inst, 261.63, duration=1.0)
            sound = pygame.sndarray.make_sound(wave)
            sound.play()
            time.sleep(1.2)
        
        pygame.mixer.quit()