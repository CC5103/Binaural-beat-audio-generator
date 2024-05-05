# 2023/8/28 ZHOU YUNHAO

import numpy as np
from scipy.io.wavfile import write
from scipy.signal import hilbert
import os

#音声ファイル保存先(フォルダパス)
folder = os.getcwd() + r"\Save_file"

initial_f1 = 43#左耳変調周波数
initial_f2 = 37#右耳変調周波数

initial_cf = 550#搬送周波数

# 初期パラメータの設定
parameters_list = [
    #(左耳変調周波数, 右耳変調周波数, 左耳変調周波数, 右耳変調周波数, ２段階目の左耳変調周波数, ２段階目の右耳変調周波数, 搬送周波数変化時間time_2, 搬送周波数変化時間time_3),
    (initial_f1 , initial_f2, initial_cf, initial_cf, initial_cf + 0.25 * (initial_f1 - initial_f2), initial_cf - 0.25 * (initial_f1 - initial_f2), 8, 10),
    (initial_f1 , initial_f2, initial_cf, initial_cf, initial_cf + 0.25 * (initial_f1 - initial_f2), initial_cf - 0.25 * (initial_f1 - initial_f2), 6, 9),
    (initial_f1 , initial_f2, initial_cf, initial_cf, initial_cf + 0.25 * (initial_f1 - initial_f2), initial_cf - 0.25 * (initial_f1 - initial_f2), 9, 11),
    (initial_f1 , initial_f2, initial_cf, initial_cf, initial_cf + 0.25 * (initial_f1 - initial_f2), initial_cf - 0.25 * (initial_f1 - initial_f2), 7, 10),
    (initial_f1 , initial_f2, initial_cf, initial_cf, initial_cf + 0.25 * (initial_f1 - initial_f2), initial_cf - 0.25 * (initial_f1 - initial_f2), 8, 11),
    (initial_f1 , initial_f2, initial_cf, initial_cf, initial_cf + 0.25 * (initial_f1 - initial_f2), initial_cf - 0.25 * (initial_f1 - initial_f2), 7, 9)
]

initial_time_1 = 3#搬送周波数変化時間time_1
time_1 = initial_time_1
time_4 = 14#音声の長さ

amplitude_1 = 1#左耳振幅
amplitude_2 = 1#右耳振幅

def limit(time, envelope_1_1, envelope_2_1, envelope_1_2, envelope_2_2, f1, f2, sample_rate=44100):
    envelope_1_1 = np.round(envelope_1_1, 3)[round((time - (1/(2*f1))) * sample_rate):round(time * sample_rate) + 1]
    envelope_1_2 = np.round(envelope_1_2, 3)[round((time - (1/(2*f1))) * sample_rate):round(time * sample_rate) + 1]
    envelope_1 = envelope_1_1 + envelope_1_2
    # print(envelope_1)
    common_values = np.where(envelope_1 == 0)[0]
    # print(common_values)
    envelope_1 = common_values[round(len(common_values)/2)]
    time_at_min_1 = t[round((time - (1/(2*f1))) * sample_rate):round(time * sample_rate) + 1][envelope_1]

    envelope_2_1 = np.round(envelope_2_1, 3)[round((time - (1/(2*f2))) * sample_rate):round(time * sample_rate) + 1]
    envelope_2_2 = np.round(envelope_2_2, 3)[round((time - (1/(2*f2))) * sample_rate):round(time * sample_rate) + 1]
    envelope_2 = envelope_2_1 + envelope_2_2
    # print(envelope_2)
    common_values = np.where(envelope_2 == 0)[0]
    # print(common_values)
    envelope_2 = common_values[round(len(common_values)/2)]
    time_at_min_2 = t[round((time - (1/(2*f2))) * sample_rate):round(time * sample_rate) + 1][envelope_2]
    return time_at_min_1, time_at_min_2

# ステレオの音声波形を生成する関数
def generate_stereo_wave(f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_2, time_3, time_4, sample_rate=44100):
    global t
    t = np.linspace(0, time_4, round(sample_rate * time_4), endpoint=False)

    f1_signal = (1 + np.sin(2 * np.pi * f1 * t)) / 2
    f2_signal = (1 + np.sin(2 * np.pi * f2 * t)) / 2

    cf1_signal_1 = amplitude_1 * np.sin(2 * np.pi * cf1_1 * t)
    cf2_signal_1 = amplitude_2 * np.sin(2 * np.pi * cf2_1 * t)

    cf1_signal_2 = amplitude_1 * np.sin(2 * np.pi * cf1_2 * t)
    cf2_signal_2 = amplitude_2 * np.sin(2 * np.pi * cf2_2 * t)

    # 包络线计算
    analytic_signal_1_1 = hilbert(cf1_signal_1 * f1_signal)
    envelope_1_1 = np.abs(analytic_signal_1_1)
    analytic_signal_1_2 = hilbert(cf1_signal_2 * f1_signal)
    envelope_1_2 = np.abs(analytic_signal_1_2)

    analytic_signal_2_1 = hilbert(cf2_signal_1 * f2_signal)
    envelope_2_1 = np.abs(analytic_signal_2_1)
    analytic_signal_2_2 = hilbert(cf2_signal_2 * f2_signal)
    envelope_2_2 = np.abs(analytic_signal_2_2)

    # 包络线極小値計算
    #time_1
    time_at_min_1_1, time_at_min_2_1 = limit(time_1, envelope_1_1, envelope_2_1, envelope_1_2, envelope_2_2, f1, f2, sample_rate=44100)

    #time_2
    time_at_min_1_2, time_at_min_2_2 = limit(time_2, envelope_1_1, envelope_2_1, envelope_1_2, envelope_2_2, f1, f2, sample_rate=44100)

    #time_3
    time_at_min_1_3, time_at_min_2_3 = limit(time_3, envelope_1_1, envelope_2_1, envelope_1_2, envelope_2_2, f1, f2, sample_rate=44100)

    new_amplitude_1 = np.concatenate([
        np.ones(round(time_at_min_1_1 * sample_rate)),
        np.zeros(round((time_at_min_1_2 - time_at_min_1_1) * sample_rate)),
        np.ones(round((time_at_min_1_3 - time_at_min_1_2) * sample_rate)),
        np.zeros(round((time_4 - time_at_min_1_3) * sample_rate))
    ])
    new_amplitude_2 = np.concatenate([
        np.zeros(round(time_at_min_1_1 * sample_rate)),
        np.ones(round((time_at_min_1_2 - time_at_min_1_1) * sample_rate)),
        np.zeros(round((time_at_min_1_3 - time_at_min_1_2) * sample_rate)),
        np.ones(round((time_4 - time_at_min_1_3) * sample_rate))
    ])

    left_channel_1 = (cf1_signal_1 * f1_signal) * new_amplitude_1
    left_channel_2 = (cf1_signal_2 * f1_signal) * new_amplitude_2

    new_amplitude_1 = np.concatenate([
        np.ones(round(time_at_min_2_1 * sample_rate)),
        np.zeros(round((time_at_min_2_2 - time_at_min_2_1) * sample_rate)),
        np.ones(round((time_at_min_2_3 - time_at_min_2_2) * sample_rate)),
        np.zeros(round((time_4 - time_at_min_2_3) * sample_rate))
    ])
    new_amplitude_2 = np.concatenate([
        np.zeros(round(time_at_min_2_1 * sample_rate)),
        np.ones(round((time_at_min_2_2 - time_at_min_2_1) * sample_rate)),
        np.zeros(round((time_at_min_2_3 - time_at_min_2_2) * sample_rate)),
        np.ones(round((time_4 - time_at_min_2_3) * sample_rate))
    ])

    right_channel_1 = (cf2_signal_1 * f2_signal) * new_amplitude_1
    right_channel_2 = (cf2_signal_2 * f2_signal) * new_amplitude_2

    return left_channel_1, left_channel_2, right_channel_1, right_channel_2

for idx, params in enumerate(parameters_list, start=1):
    f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, time_2, time_3 = params

    # ステレオ波形の生成
    left_channel_1, left_channel_2, right_channel_1, right_channel_2 = generate_stereo_wave(f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_2, time_3, time_4)

    combined_lift = left_channel_1 + left_channel_2
    combined_right = right_channel_1 + right_channel_2
    stereo_wave_1 = np.column_stack((combined_lift, combined_right))
    combined_audio = np.array(stereo_wave_1 * (2 ** 15 - 1), dtype=np.int16)

    time_2 = round(time_2)
    time_3 = round(time_3)

    # Save the audio file
    file_name = f"{idx}：[{time_2},{time_3}][{f1},{f2}];[{cf1_1},{cf2_1}];[{cf1_2},{cf2_2}].wav"
    write(f"{folder}\\{file_name}", 44100, combined_audio)

    print(f"Saved audio file {file_name}")

print("All audio files saved successfully!")