#!/usr/bin/env python3
"""
マリンバ両手弾き - pygame版
より安定した動作で和音演奏が可能
"""

import numpy as np
import pygame
import time
import threading
from collections import defaultdict

# 音階の周波数
NOTES = {
    # 第3オクターブ
    'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'F3': 174.61,
    'G3': 196.00, 'A3': 220.00, 'B3': 246.94,
    # 第4オクターブ
    'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
    'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25,
}

# 和音定義
CHORDS = {
    'C': ['C4', 'E4', 'G4'],
    'Dm': ['D4', 'F4', 'A4'],
    'Em': ['E4', 'G4', 'B4'],
    'F': ['F4', 'A4', 'C5'],
    'G': ['G4', 'B4', 'D4'],
    'Am': ['A4', 'C5', 'E4'],
    'G7': ['G3', 'B3', 'D4', 'F4'],
}


class MarimbaSoundBank:
    """マリンバ音源バンク（事前生成して高速再生）"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.sounds = {}
        self._generate_all_sounds()
    
    def _generate_marimba_wave(self, frequency, duration=2.0):
        """マリンバ波形を生成"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # マリンバの音色（基音＋倍音）
        fundamental = np.sin(2 * np.pi * frequency * t)
        harmonic2 = 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        harmonic3 = 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
        harmonic4 = 0.05 * np.sin(2 * np.pi * frequency * 4 * t)
        
        wave = fundamental + harmonic2 + harmonic3 + harmonic4
        
        # エンベロープ（マリンバ特有の減衰）
        attack_time = 0.005
        decay_time = 0.15
        
        envelope = np.ones_like(t)
        attack_samples = int(attack_time * self.sample_rate)
        decay_samples = int(decay_time * self.sample_rate)
        
        # アタック
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # ディケイ〜サステイン
        decay_end = attack_samples + decay_samples
        if decay_end < len(envelope):
            envelope[attack_samples:decay_end] = np.linspace(1, 0.2, decay_samples)
            # 指数減衰
            remaining = len(envelope) - decay_end
            envelope[decay_end:] = 0.2 * np.exp(-2 * np.linspace(0, 4, remaining))
        
        # 適用
        wave = wave * envelope * 0.3
        
        # 16ビット整数に変換
        wave = (wave * 32767).astype(np.int16)
        
        # ステレオ化
        stereo = np.zeros((len(wave), 2), dtype=np.int16)
        stereo[:, 0] = wave
        stereo[:, 1] = wave
        
        return stereo
    
    def _generate_all_sounds(self):
        """全音階の音を事前生成"""
        print("音源を生成中...", end='', flush=True)
        for note, freq in NOTES.items():
            wave = self._generate_marimba_wave(freq)
            self.sounds[note] = pygame.sndarray.make_sound(wave)
        print(" 完了！")
    
    def get_sound(self, note):
        """音源を取得"""
        return self.sounds.get(note)


class PolyphonicMarimba:
    """ポリフォニック・マリンバ（pygame版）"""
    
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(32)  # 同時発音数を増やす
        
        self.sound_bank = MarimbaSoundBank()
        self.active_channels = []
    
    def play_note(self, note, velocity=0.7):
        """単音を再生"""
        sound = self.sound_bank.get_sound(note)
        if sound:
            # 空いているチャンネルを探す
            channel = pygame.mixer.find_channel()
            if channel:
                channel.set_volume(velocity)
                channel.play(sound)
                self.active_channels.append(channel)
    
    def play_chord(self, notes, velocity=0.7):
        """和音を再生（同時に複数音）"""
        for note in notes:
            self.play_note(note, velocity)
    
    def play_with_timing(self, notes, delays, velocity=0.7):
        """タイミングをずらして和音を再生（ストラム効果）"""
        for i, note in enumerate(notes):
            if i > 0:
                time.sleep(delays)
            self.play_note(note, velocity)
    
    def stop_all(self):
        """全ての音を停止"""
        pygame.mixer.stop()
        self.active_channels.clear()


def demo_pygame_both_hands():
    """pygame版の両手弾きデモ"""
    print("=== マリンバ両手弾きデモ（pygame版） ===\n")
    
    marimba = PolyphonicMarimba()
    
    # 1. 和音のデモ
    print("1. 基本的な和音")
    for chord_name, notes in [('C', CHORDS['C']), ('F', CHORDS['F']), ('G', CHORDS['G'])]:
        print(f"   {chord_name}: {' '.join(notes)}")
        marimba.play_chord(notes)
        time.sleep(1.5)
    
    # 2. アルペジオ
    print("\n2. アルペジオ（分散和音）")
    for _ in range(2):
        for note in ['C4', 'E4', 'G4', 'C5']:
            marimba.play_note(note, velocity=0.5)
            time.sleep(0.12)
    time.sleep(0.5)
    
    # 3. ストラム効果
    print("\n3. ギターストラム風")
    marimba.play_with_timing(CHORDS['C'], delays=0.02)
    time.sleep(0.5)
    marimba.play_with_timing(CHORDS['G7'], delays=0.02)
    time.sleep(1.0)
    
    # 4. 両手パターン
    print("\n4. 左手ベース + 右手メロディ")
    
    def play_bass_line():
        """ベースライン（左手）"""
        bass_pattern = ['C3', 'C3', 'G3', 'G3', 'A3', 'A3', 'F3', 'G3']
        for note in bass_pattern:
            marimba.play_note(note, velocity=0.4)
            time.sleep(0.4)
    
    def play_melody_line():
        """メロディ（右手）"""
        time.sleep(0.05)  # 微妙にずらす
        melody = ['E4', 'D4', 'C4', 'D4', 'E4', 'E4', 'E4', '-',
                  'D4', 'D4', 'D4', '-', 'E4', 'G4', 'G4', '-']
        for note in melody[:8]:
            if note != '-':
                marimba.play_note(note, velocity=0.6)
            time.sleep(0.4)
    
    # 両手同時演奏
    bass_thread = threading.Thread(target=play_bass_line)
    melody_thread = threading.Thread(target=play_melody_line)
    
    bass_thread.start()
    melody_thread.start()
    
    bass_thread.join()
    melody_thread.join()
    
    # 5. リッチな和音進行
    print("\n5. ジャズ風コード進行")
    progression = [
        (['C4', 'E4', 'G4', 'B4'], "Cmaj7"),
        (['A3', 'C4', 'E4', 'G4'], "Am7"),
        (['D3', 'F3', 'A3', 'C4'], "Dm7"),
        (['G3', 'B3', 'D4', 'F4'], "G7"),
    ]
    
    for notes, name in progression * 2:
        print(f"   {name}")
        marimba.play_chord(notes, velocity=0.5)
        time.sleep(0.8)
    
    # 6. 高速アルペジオ
    print("\n6. 高速アルペジオ")
    arpeggio_notes = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4']
    for _ in range(4):
        for note in arpeggio_notes:
            marimba.play_note(note, velocity=0.3)
            time.sleep(0.05)
    
    # 終了和音
    print("\n終了！")
    marimba.play_chord(['C3', 'C4', 'E4', 'G4', 'C5'], velocity=0.8)
    time.sleep(2.0)
    
    pygame.mixer.quit()


def simple_piano_mode():
    """シンプルなピアノモード（キーボードで演奏）"""
    print("\n=== キーボード演奏モード ===")
    print("キーマッピング:")
    print("  A S D F G H J K L → C D E F G A B C")
    print("  Q W E R T Y U I → C# D# F# G# A#")
    print("  Z: 低いC, X: 低いG")
    print("  Space: サステイン")
    print("  ESC: 終了")
    
    # キーと音のマッピング
    key_map = {
        pygame.K_z: 'C3', pygame.K_x: 'G3',
        pygame.K_a: 'C4', pygame.K_s: 'D4', pygame.K_d: 'E4',
        pygame.K_f: 'F4', pygame.K_g: 'G4', pygame.K_h: 'A4',
        pygame.K_j: 'B4', pygame.K_k: 'C5', pygame.K_l: 'D5',
        pygame.K_q: 'C#4', pygame.K_w: 'D#4', pygame.K_e: 'F#4',
        pygame.K_r: 'G#4', pygame.K_t: 'A#4',
    }
    
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("マリンバ・キーボード")
    
    marimba = PolyphonicMarimba()
    pressed_keys = set()
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in key_map and event.key not in pressed_keys:
                    note = key_map[event.key]
                    print(f"♪ {note}")
                    marimba.play_note(note)
                    pressed_keys.add(event.key)
            
            elif event.type == pygame.KEYUP:
                if event.key in pressed_keys:
                    pressed_keys.remove(event.key)
        
        # 画面更新
        screen.fill((30, 30, 30))
        text = pygame.font.Font(None, 24).render(
            f"演奏中... 押されているキー: {len(pressed_keys)}", 
            True, (255, 255, 255)
        )
        screen.blit(text, (50, 80))
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '-k':
        # キーボードモード
        simple_piano_mode()
    else:
        # デモモード
        demo_pygame_both_hands()