import moviepy.editor as mp
from pydub import AudioSegment
import speech_recognition as sr
import os
from tqdm import tqdm

def extract_audio_from_video(video_path, audio_path):
    """从视频文件中提取音频并保存为 MP3 文件"""
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def convert_audio_to_wav(input_audio_path, output_wav_path):
    """将音频文件转换为 WAV 格式"""
    audio = AudioSegment.from_file(input_audio_path)
    audio.export(output_wav_path, format="wav")

def transcribe_audio(wav_path):
    """将 WAV 音频文件转换为文本"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="zh-CN")  # 使用中文识别
            return text
        except sr.UnknownValueError:
            return "无法识别音频"
        except sr.RequestError as e:
            return f"请求错误: {e}"

def main():
    video_path = input("请输入 MP4 文件路径：")
    audio_path = "extracted_audio.mp3"
    wav_path = "extracted_audio.wav"

    # 提取音频
    print("正在提取音频...")
    for _ in tqdm(range(1), desc="提取音频"):
        extract_audio_from_video(video_path, audio_path)
    print(f"音频已提取并保存到 {audio_path}")

    # 转换音频格式为 WAV
    print("正在转换音频格式为 WAV...")
    for _ in tqdm(range(1), desc="转换音频"):
        convert_audio_to_wav(audio_path, wav_path)
    print(f"音频已转换为 WAV 格式并保存到 {wav_path}")

    # 转换音频为文本
    print("正在转换音频为文本...")
    for _ in tqdm(range(1), desc="转换文本"):
        text = transcribe_audio(wav_path)
    print("转换后的文本内容如下：")
    print(text)

    # 清理临时文件
    os.remove(audio_path)
    os.remove(wav_path)

if __name__ == "__main__":
    main()