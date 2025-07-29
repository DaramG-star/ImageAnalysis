import pyaudio
import numpy as np
import librosa
import tkinter as tk
from threading import Thread, Lock
import scipy.signal as signal

RATE = 44100
CHUNK = 1024
p = pyaudio.PyAudio()
lock = Lock()

# 🔹 장치 목록 가져오기
input_devices = []
output_devices = []

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get("maxInputChannels") > 0:
        input_devices.append((i, dev["name"]))
    if dev.get("maxOutputChannels") > 0:
        output_devices.append((i, dev["name"]))

selected_input_index = input_devices[0][0]
selected_output_index = output_devices[0][0]

# 🔹 기본 입력/출력 스트림
stream_in = p.open(format=pyaudio.paFloat32,
                   channels=1,
                   rate=RATE,
                   input=True,
                   input_device_index=selected_input_index,
                   frames_per_buffer=CHUNK)

stream_out = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=RATE,
                    output=True,
                    output_device_index=selected_output_index,
                    frames_per_buffer=CHUNK)

# ---------------- 필터 ----------------
def pitch_shift(audio, steps):
    return librosa.effects.pitch_shift(y=audio, sr=RATE, n_steps=steps)

def robot_voice(audio):
    return audio * np.sign(np.sin(2*np.pi*100*np.arange(len(audio))/RATE))

def megaphone(audio):
    b, a = signal.butter(2, [1000/(RATE/2), 4000/(RATE/2)], btype='band')
    return signal.lfilter(b, a, audio)

def telephone(audio):
    b, a = signal.butter(2, [300/(RATE/2), 3400/(RATE/2)], btype='band')
    return signal.lfilter(b, a, audio)

def echo(audio):
    delay = int(0.2 * RATE)
    echo_buf = np.zeros(len(audio) + delay)
    echo_buf[:len(audio)] = audio
    echo_buf[delay:] += 0.6 * audio
    return echo_buf[:len(audio)]

# ---------------- 필터 상태 ----------------
selected_filters = {
    "Pitch Up": False,
    "Pitch Down": False,
    "Robot": False,
    "Megaphone": False,
    "Telephone": False,
    "Echo": False
}

def toggle_filter(name):
    selected_filters[name] = not selected_filters[name]
    buttons[name].config(bg="green" if selected_filters[name] else "gray")

# 🔹 입력 장치 변경
def change_input_device(event=None):
    global stream_in, selected_input_index
    sel_name = input_var.get()
    for idx, name in input_devices:
        if name == sel_name:
            selected_input_index = idx
            break
    with lock:
        stream_in.stop_stream()
        stream_in.close()
        stream_in = p.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=RATE,
                           input=True,
                           input_device_index=selected_input_index,
                           frames_per_buffer=CHUNK)

# 🔹 출력 장치 변경
def change_output_device(event=None):
    global stream_out, selected_output_index
    sel_name = output_var.get()
    for idx, name in output_devices:
        if name == sel_name:
            selected_output_index = idx
            break
    with lock:
        stream_out.stop_stream()
        stream_out.close()
        stream_out = p.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=RATE,
                            output=True,
                            output_device_index=selected_output_index,
                            frames_per_buffer=CHUNK)

# 🔹 오디오 처리 스레드
def process_audio():
    global stream_out
    while True:
        data = stream_in.read(CHUNK, exception_on_overflow=False)
        audio = np.frombuffer(data, dtype=np.float32)

        if selected_filters["Pitch Up"]:
            audio = pitch_shift(audio, 3)
        if selected_filters["Pitch Down"]:
            audio = pitch_shift(audio, -3)
        if selected_filters["Robot"]:
            audio = robot_voice(audio)
        if selected_filters["Megaphone"]:
            audio = megaphone(audio)
        if selected_filters["Telephone"]:
            audio = telephone(audio)
        if selected_filters["Echo"]:
            audio = echo(audio)

        if len(audio) < CHUNK:
            audio = np.pad(audio, (0, CHUNK-len(audio)))
        elif len(audio) > CHUNK:
            audio = audio[:CHUNK]

        audio = audio * 1.5  # 볼륨 보정

        with lock:
            if stream_out.is_active():
                stream_out.write(audio.astype(np.float32).tobytes())

# ---------------- GUI ----------------
root = tk.Tk()
root.title("🎤 실시간 보이스 체인저")
root.geometry("400x500")

# 입력 장치 선택
tk.Label(root, text="입력 장치 선택 (마이크)", font=("Arial", 12)).pack(pady=5)
input_var = tk.StringVar()
input_var.set(input_devices[0][1])
input_menu = tk.OptionMenu(root, input_var, *[name for _, name in input_devices], command=change_input_device)
input_menu.pack(pady=5)

# 출력 장치 선택
tk.Label(root, text="출력 장치 선택 (스피커)", font=("Arial", 12)).pack(pady=5)
output_var = tk.StringVar()
output_var.set(output_devices[0][1])
output_menu = tk.OptionMenu(root, output_var, *[name for _, name in output_devices], command=change_output_device)
output_menu.pack(pady=5)

# 필터 버튼
buttons = {}
for name in selected_filters.keys():
    btn = tk.Button(root, text=name, width=20, height=2,
                    bg="gray", fg="white",
                    command=lambda n=name: toggle_filter(n))
    btn.pack(pady=5)
    buttons[name] = btn

thread = Thread(target=process_audio, daemon=True)
thread.start()

root.mainloop()
