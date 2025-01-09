import queue
import struct
import sys
import math
import wave
from typing import NamedTuple
import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline
import time

class VadConfig(NamedTuple):
    """発話区間検出を設定するクラス.
    Attributes:
        threshold (int): 発話区間検出を判定するパワーのしきい値 (dB)
        vad_start (float): 発話区間を開始判定する秒数（sec）
        vad_end (float): 発話区間を終了判定する秒数 (sec)
    """
    threshold: int = 45
    vad_start: float = 0.3
    vad_end: float = 1.0

class VoiceActivityDetector:
    """マイク入力から発話区間を検出して保存するクラス."""
    def __init__(self, rate: int, chunk: int, vad_config: VadConfig):
        """音声入力ストリームを初期化する.
        Args:
            rate (int): サンプリングレート (Hz)
            chunk (int): 音声データを受け取る単位（サンプル数）
            vad_config (VadConfig): 発話区間検出の設定
        """
        # マイク入力のパラメータ
        self.rate = rate
        self.chunk = chunk
        # 入力された音声データを保持するキュー
        self.buff = queue.Queue()
        # 発話区間の音声データを保持するリスト
        self.voice_data = []
        # 発話区間検出のパラメータ
        self.vad_config = {
            "threshold": vad_config.threshold,
            "vad_start": vad_config.vad_start,
            "vad_end": vad_config.vad_end,
        }
        # 発話区間検出の作業用変数
        self.workspace = {
            "is_speaking": False,  # 現在発話区間を認定しているか
            "count_on": 0,  # 現在まででしきい値以上の区間が連続している数
            "count_off": 0,  # 現在まででしきい値以下の区間が連続している数
            "voice_end": False,  # 発話が終了したか
            "result": None,  # 発話区間の音声認識結果
        }
        # マイク音声入力の初期化
        self.input_stream = None
        # STT setup
        model_id = "kotoba-tech/kotoba-whisper-v2.0"
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"attn_implementation": "sdpa"} if torch.cuda.is_available() else {}
        self.generate_kwargs = {"language": "ja", "task": "transcribe"}
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=device,
            model_kwargs=model_kwargs
        )

    def callback(self, indata, frames, time, status):
        """音声入力の度に呼び出される関数.
        音声パワーに基づいて発話区間を判定し、発話区間のデータを保存する.
        """
        if status:
            print(status, file=sys.stderr)
        audio_data = bytes(indata) # 音声データをバイト列に変換
        indata2 = struct.unpack(f"{len(indata) / 2:.0f}h", audio_data) # 音声のパワー計算
        rms = math.sqrt(np.square(indata2).mean())
        power = 20 * math.log10(rms) if rms > 0.0 else -math.inf
        if (power >= self.vad_config["threshold"] and not self.workspace["is_speaking"]): # 発話区間の開始検出
            self.workspace["count_on"] += 1
            count_on_sec = float(self.workspace["count_on"] * self.chunk) / self.rate
            if count_on_sec >= self.vad_config["vad_start"]:
                print("発話開始", flush=True)
                self.workspace["is_speaking"] = True
                self.workspace["count_on"] = 0
                self.voice_data = []  # 新しい発話区間の開始
        if self.workspace["is_speaking"]:
            self.voice_data.extend(indata2)
        if power < self.vad_config["threshold"] and self.workspace["is_speaking"]: # 発話区間の終了検出
            self.workspace["count_off"] += 1
            count_off_sec = float(self.workspace["count_off"] * self.chunk) / self.rate
            if count_off_sec >= self.vad_config["vad_end"]:
                print("発話終了", flush=True)
                self.workspace["voice_end"] = True
                self.workspace["is_speaking"] = False
                self.workspace["count_off"] = 0
                self.workspace["result"] = self.voice2text()  # 発話区間を保存
        if power >= self.vad_config["threshold"]: # カウンタのリセット
            self.workspace["count_off"] = 0
        else:
            self.workspace["count_on"] = 0

    def voice2text(self):
        if not self.voice_data:
            return
        filename = "voice.wav"
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)  # モノラル
            wf.setsampwidth(2)  # 16bit
            wf.setframerate(self.rate)
            wf.writeframes(np.array(self.voice_data).astype(np.int16).tobytes())
        result = self.pipe(filename, generate_kwargs=self.generate_kwargs)
        self.voice_data = []
        if len(result["text"]) > 4: # result["text"]が4文字以下の場合は認識失敗とみなす
            return result["text"]
        else:
            return None

    def start_recording(self):
        """マイク入力の開始"""
        self.input_stream = sd.RawInputStream(
            samplerate=self.rate,
            blocksize=self.chunk,
            dtype="int16",
            channels=1,
            callback=self.callback
        )
        self.input_stream.start()

    def stop_recording(self):
        """マイク入力の停止"""
        if self.input_stream is not None:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None

def record_voice(duration: float = 10.0,
                chunk_size: int = 8000,
                threshold: int = 40, # 40
                vad_start: float = 0.3,
                vad_end: float = 0.5):
    """指定した時間だけマイク入力から発話を検出して保存する.
    Args:
        duration (float): 録音する時間（秒）
        chunk_size (int): 音声データを受け取る単位（サンプル数）
        threshold (int): 発話区間検出を判定するパワーのしきい値 (dB)
        vad_start (float): 発話区間を開始判定する秒数（sec）
        vad_end (float): 発話区間を終了判定する秒数 (sec）
    """
    # 入力デバイス情報からサンプリング周波数を取得
    input_device_info = sd.query_devices(kind="input")
    sample_rate = int(input_device_info["default_samplerate"])
    # 発話区間検出の設定
    vad_config = VadConfig(threshold, vad_start, vad_end)
    # 音声検出器の初期化
    detector = VoiceActivityDetector(sample_rate, chunk_size, vad_config)
    print("録音開始...")
    start_time = time.time()
    detector.start_recording()
    # detector.workspace["result"]がNoneでなくなるまで待機
    while detector.workspace["result"] is None and time.time()-start_time < duration:
        sd.sleep(100)
    detector.stop_recording()
    print("録音終了", time.time()-start_time)
    return detector.workspace["result"]
if __name__ == "__main__":
    result = record_voice(duration=10.0)
    print(result)