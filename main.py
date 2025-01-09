from openai import OpenAI
import sys,io
import json
import requests
import simpleaudio
import time
import wave
import queue
import struct
import math
from typing import NamedTuple
import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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

class Speaker:
    def __init__(self, memory_size=5):
        """
        memory_size: 何回分の対話を保存するか
        例えば5ならシステムプロンプトも含めて11個のメッセージが保存される
        """
        # self.model="gpt-4o"
        self.model="ft:gpt-4o-2024-08-06:personal:balck-joke:A6g8Gkxj"
        self.client = OpenAI(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxx")
        self.system_conf = "You are a good friend and you likes black jokes."
        self.messages = [{"role": "system", "content": self.system_conf}]
        self.memory_size = memory_size

    def chatProcess(self, userMessage, forget, url=False):
        """
        ChatGPTによる応答生成
        """
        if forget:
            self.messages = [{"role": "system", "content": self.system_conf}]
        if len(self.messages) > (self.memory_size+1)*2:
            del self.messages[1:3]
        if url:
            self.messages += [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": userMessage}}]}]
            return "画像を受け取りました。"
        self.messages += [{"role": "user", "content": userMessage}]
        response = self.client.chat.completions.create(model=self.model,messages=self.messages)
        ai_response = response.choices[0].message.content
        self.messages += [{"role": "assistant", "content": ai_response}]
        return ai_response

    def play_wav_safely(self, wav_data):
        """
        WAVファイルを途切れることなく再生する関数
        """
        # 先頭に無音データを追加
        with wave.open(io.BytesIO(wav_data), 'rb') as wave_file:
            n_channels = wave_file.getnchannels()
            sampwidth = wave_file.getsampwidth()
            framerate = wave_file.getframerate()
            n_frames = wave_file.getnframes()
            wave_data = wave_file.readframes(n_frames)
        sec = 0.5
        padding_frames = int(sec * framerate)
        padding = b'\x00' * (padding_frames * n_channels * sampwidth)
        padded_data = padding + wave_data
        wave_obj = simpleaudio.WaveObject(padded_data, n_channels, sampwidth, framerate)
        play_obj = wave_obj.play()
        return play_obj

    def text2voice(self, text, speaker=888753765):
        """
        テキストから音声を生成する関数
        """
        host = 'localhost'
        port = 10101
        params = (
            ('text', text),
            ('speaker', speaker),
        )
        response1 = requests.post(
            f'http://{host}:{port}/audio_query',
            params=params
        )
        if response1.status_code != 200:
            print('get audio_query failed')
            print(response1)
            return
        headers = {
            'Content-Type': 'application/json',
            'accept': 'audio/wav'}
        response2 = requests.post(
            f'http://{host}:{port}/synthesis',
            headers=headers,
            params=params,
            data=json.dumps(response1.json())
        )
        if response2.status_code != 200:
            print('get synthesis failed')
            print(response2)
            return
        print("AI: ", text, flush=True)
        play_obj = self.play_wav_safely(response2.content)
        play_obj.wait_done()

    def get_speaker_id(self):
        response = requests.get('http://localhost:10101/speakers')
        if response.status_code != 200:
            print('get speakers failed')
            print(response)
            return
        else:
            ids = []
            for speaker in response.json():
                print(speaker['name'])
                for style in speaker['styles']:
                    print('    ' + style['name'] + ' ' + str(style['id']))
                    ids.append(style['id'])
            return ids

class Listener:
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
                self.workspace["is_speaking"] = True
                self.workspace["count_on"] = 0
                self.voice_data = []  # 新しい発話区間の開始
        if self.workspace["is_speaking"]:
            self.voice_data.extend(indata2)
        if power < self.vad_config["threshold"] and self.workspace["is_speaking"]: # 発話区間の終了検出
            self.workspace["count_off"] += 1
            count_off_sec = float(self.workspace["count_off"] * self.chunk) / self.rate
            if count_off_sec >= self.vad_config["vad_end"]:
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
        if len(result["text"]) > 4 and result["text"] != "ありがとうございました": # result["text"]が4文字以下の場合は認識失敗とみなす
            print("あなた: ", result["text"], flush=True)
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

def main():
    # マイク入力の設定
    chunk_size = 8000
    threshold = 40
    vad_start = 0.3
    vad_end = 0.5
    # Listenerの初期化
    input_device_info = sd.query_devices(kind="input")
    print("入力デバイス情報:", input_device_info)
    sample_rate = int(input_device_info["default_samplerate"])
    vad_config = VadConfig(threshold, vad_start, vad_end)
    listener = Listener(sample_rate, chunk_size, vad_config)
    # Speakerの初期化
    speaker = Speaker()
    print(speaker.get_speaker_id())
    id = 888753765
    # メインループ
    while True:
        print("録音開始", flush=True)
        listener.workspace["result"] = None
        listener.start_recording()
        while listener.workspace["result"] is None:
            sd.sleep(100)
        listener.stop_recording()
        response = speaker.chatProcess(listener.workspace["result"], False)
        speaker.text2voice(response, id)

if __name__ == "__main__":
    main()