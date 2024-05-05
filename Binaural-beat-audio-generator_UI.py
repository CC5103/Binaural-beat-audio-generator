#2023/8/28 ZHOU YUNHAO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from scipy.io.wavfile import write
import simpleaudio as sa
import threading
import time as wait_time
from scipy.signal import hilbert
import os

#音声ファイル保存先(フォルダパス)
folder = os.getcwd() + r"\Save_file"

# 初期パラメータの設定
initial_f1 = 39#左耳変調周波数
initial_f2 = 45#右耳変調周波数

initial_cf = 500#搬送周波数
initial_cf1_1 = 500
initial_cf2_1 = 500
initial_cf1_2 = 498.5
initial_cf2_2 = 501.5

initial_time_1 = 3
initial_time_2 = 7
initial_time_3 = 10
initial_time_4 = 14

initial_amplitude_1 = 1
initial_amplitude_2 = 1

time_1 = initial_time_1
time_2 = initial_time_2
time_3 = initial_time_3
time_4 = initial_time_4

f1_ = 0
f2_ = 0
cf1_1_ = 0
cf2_1_ = 0
cf1_2_ = 0
cf2_2_ = 0
amplitude_1_ = 0
amplitude_2_ = 0
time_1_ = 0
time_4_ = 0
time_1__ = initial_time_1
time_4__ = initial_time_4
cf_ = initial_cf
amplitude_1 = initial_amplitude_1
amplitude_2 = initial_amplitude_2

combined_audio_obj = None # 音声再生オブジェクトの初期化
num = 0

def graph(left_channel_1, left_channel_2, right_channel_1, right_channel_2, time_4):
    t = np.linspace(0, time_4, round(44100 * time_4), endpoint=False)

    # グラフのクリア
    ax_left.clear()
    ax_right.clear()
    ax_beating.clear()

    # 左右チャンネルの波形をプロット
    combined_lift = left_channel_1 + left_channel_2
    ax_left.plot(t, combined_lift, color='blue')
    ax_left.set_ylim(-2, +2)
    ax_left.set_ylabel('Amplitude_lift')

    combined_right = right_channel_1 + right_channel_2
    ax_right.plot(t, combined_right, color='red')
    ax_right.set_ylim(-2, +2)
    ax_right.set_ylabel('Amplitude_right')

    # ビート音の波形をプロット
    ax_beating.plot(t, (combined_lift - combined_right), color='green')
    ax_beating.set_ylim(-2, +2)
    ax_beating.set_xlabel('Time (s)')
    ax_beating.set_ylabel('Amplitude_beating')

#包絡線の谷時間計算
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

    # 包络线极小值
    #time_1
    time_at_min_1_1, time_at_min_2_1 = limit(time_1, envelope_1_1, envelope_2_1, envelope_1_2, envelope_2_2, f1, f2, sample_rate=44100)
    print(time_at_min_1_1, time_at_min_2_1)

    #time_2
    time_at_min_1_2, time_at_min_2_2 = limit(time_2, envelope_1_1, envelope_2_1, envelope_1_2, envelope_2_2, f1, f2, sample_rate=44100)
    print(time_at_min_1_2, time_at_min_2_2)

    #time_3
    time_at_min_1_3, time_at_min_2_3 = limit(time_3, envelope_1_1, envelope_2_1, envelope_1_2, envelope_2_2, f1, f2, sample_rate=44100)
    print(time_at_min_1_3, time_at_min_2_3)

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
# 周波数や振幅の設定が変更されたときに波形を更新する関数
def update_beating_effect_1(val):
    global f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, t, time_1, time_2, time_3, time_4, combined_audio, cf, cf_
    cf = float(cf_textbox.text)
    cf_ = cf
    if amplitude_1 == 0 or amplitude_2 == 0:
        update_beating_effect_2(None)
    else:
        f1 = int(f1_slider.val)
        f2 = int(f2_slider.val)
        cf1_1 = float(cf)
        cf1_1_slider.set_val(cf1_1)
        cf2_1 = float(cf)
        cf2_1_slider.set_val(cf2_1)
        cf1_2 = float(cf + 0.25 * (f1 - f2))
        cf1_2_slider.set_val(cf1_2)
        cf2_2 = float(cf - 0.25 * (f1 - f2))
        cf2_2_slider.set_val(cf2_2)

def update_beating_effect_2(val):
    global f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, t, time_1, time_2, time_3, time_4, combined_audio, cf, cf_
    cf = float(cf_textbox.text)
    if cf != cf_:
        cf1_1 = float(cf1_1_slider.val - cf_ + cf)
        cf2_1 = float(cf2_1_slider.val - cf_ + cf)
        cf1_2 = float(cf1_2_slider.val - cf_ + cf)
        cf2_2 = float(cf2_2_slider.val - cf_ + cf)
        cf_ = cf
    else:
        cf1_1 = float(cf1_1_slider.val)
        cf2_1 = float(cf2_1_slider.val)
        cf1_2 = float(cf1_2_slider.val)
        cf2_2 = float(cf2_2_slider.val)
    f1 = int(f1_slider.val)
    f2 = int(f2_slider.val)
    amplitude_1 = amplitude_slider_1.val
    amplitude_2 = amplitude_slider_2.val
    time_1 = float(time_1_textbox.text)
    time_2 = initial_time_2
    time_3 = initial_time_3
    time_4 = float(time_4_textbox.text)

    # ステレオ波形の生成
    left_channel_1, left_channel_2, right_channel_1, right_channel_2 = generate_stereo_wave(f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_2, time_3, time_4)

    graph(left_channel_1, left_channel_2, right_channel_1, right_channel_2, time_4)
    # グラフを更新
    fig.canvas.draw()

# パラメータの変更を反映させる関数
def update_parameters(event):
    global num, cf_
    f1 = float(f1_textbox.text)
    f2 = float(f2_textbox.text)
    cf1_1 = float(cf1_1_textbox.text)
    cf2_1 = float(cf2_1_textbox.text)
    cf1_2 = float(cf1_2_textbox.text)
    cf2_2 = float(cf2_2_textbox.text)
    amplitude_1 = float(amplitude_textbox_1.text)
    amplitude_2 = float(amplitude_textbox_2.text)

    cf_ = int((cf1_1 + cf2_1)/2)
    cf_textbox.set_val(cf_)

    f1_slider.set_val(f1)
    f2_slider.set_val(f2)
    cf1_1_slider.set_val(cf1_1)
    cf2_1_slider.set_val(cf2_1)
    cf1_2_slider.set_val(cf1_2)
    cf2_2_slider.set_val(cf2_2)
    amplitude_slider_1.set_val(amplitude_1)
    amplitude_slider_2.set_val(amplitude_2)

    print("Confirm success!")
    if num % 2 == 0:
        play_status_text.set_val("Confirm success")
        play_status_text_ax.set_facecolor('gray')
    else:
        play_status_text.set_val("Confirm success")
        play_status_text_ax.set_facecolor('brown')
    num += 1
    fig.canvas.draw()

    update_beating_effect_2(None)

# 音声ファイルを保存する関数
def save_audio_file(event):
    global combined_audio_obj, combined_audio, f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, cf, cf_
    cf = float(cf_textbox.text)
    cf_m = cf_ + cf
    if cf != cf_:
        cf1_1 = float(abs(cf1_1_slider.val - cf_m))
        cf1_1_slider.set_val(cf1_1)
        cf2_1 = float(abs(cf2_1_slider.val - cf_m))
        cf2_1_slider.set_val(cf2_1)
        cf1_2 = float(abs(cf1_2_slider.val - cf_m))
        cf1_2_slider.set_val(cf1_2)
        cf2_2 = float(abs(cf2_2_slider.val - cf_m))
        cf2_2_slider.set_val(cf2_2)
        cf_ = cf
    else:
        cf1_1 = float(cf1_1_slider.val)
        cf2_1 = float(cf2_1_slider.val)
        cf1_2 = float(cf1_2_slider.val)
        cf2_2 = float(cf2_2_slider.val)
    f1 = float(f1_slider.val)
    f2 = float(f2_slider.val)
    amplitude_1 = float(amplitude_slider_1.val)
    amplitude_2 = float(amplitude_slider_2.val)
    time_1 = float(time_1_textbox.text)
    time_2 = initial_time_2
    time_3 = initial_time_3
    time_4 = float(time_4_textbox.text)
    left_channel_1, left_channel_2, right_channel_1, right_channel_2 = generate_stereo_wave(f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_2, time_3, time_4)
    combined_lift = left_channel_1 + left_channel_2
    combined_right = right_channel_1 + right_channel_2
    stereo_wave_1 = np.column_stack((combined_lift, combined_right))
    combined_audio = np.array(stereo_wave_1 * (2 ** 15 - 1), dtype=np.int16)

    if combined_audio_obj != None:
        print("Sound stop!")
        combined_audio_obj.stop()
        combined_audio_obj = None
    sample_rate = 44100
    write(f"{folder}/[{f1},{f2}];[{cf1_1},{cf2_1}];[{cf1_2},{cf2_2}];[{amplitude_1},{amplitude_2}].wav", sample_rate, combined_audio)
    print(time_1)
    print(time_2)
    print(time_3)
    print("Sound save success!")
    print(f"保存先：{folder}")
    wait_time.sleep(0.3)
    play_status_text.set_val("Save success")
    play_status_text_ax.set_facecolor('yellow')
    fig.canvas.draw()

def play(combined_audio):
    global combined_audio_obj
    print("Sound play success!")
    play_status_text_ax.set_facecolor('green')
    combined_audio_obj = sa.WaveObject(combined_audio , 2, 2, 44100)
    combined_audio_obj = combined_audio_obj.play()
    combined_audio_obj.wait_done()
    play_status_text.set_val("Not Playing")
    play_status_text_ax.set_facecolor('red')
    fig.canvas.draw()
    print("Sound play finished!")
    combined_audio_obj= None

def audio_play(event):
    global combined_audio_obj, stereo_wave_1, f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, time_1, time_2, time_3, time_4, cf, cf_, time_1_, time_4_, f1_, f2_, cf1_1_, cf2_1_ , cf1_2_, cf2_2_, amplitude_1_, amplitude_2_ , time_1_ , time_4_, time_1__, time_4__
    if combined_audio_obj != None:
        print("Sound stop!")
        combined_audio_obj.stop()
        combined_audio_obj = None
    else:
        cf = float(cf_textbox.text)
        cf_m = cf_ + cf
        if cf != cf_:
            cf1_1 = float(abs(cf1_1_slider.val - cf_m))
            cf1_1_slider.set_val(cf1_1)
            cf2_1 = float(abs(cf2_1_slider.val - cf_m))
            cf2_1_slider.set_val(cf2_1)
            cf1_2 = float(abs(cf1_2_slider.val - cf_m))
            cf1_2_slider.set_val(cf1_2)
            cf2_2 = float(abs(cf2_2_slider.val - cf_m))
            cf2_2_slider.set_val(cf2_2)
            cf_ = cf
        else:
            cf1_1 = float(cf1_1_slider.val)
            cf2_1 = float(cf2_1_slider.val)
            cf1_2 = float(cf1_2_slider.val)
            cf2_2 = float(cf2_2_slider.val)
        f1 = float(f1_slider.val)
        f2 = float(f2_slider.val)
        amplitude_1 = float(amplitude_slider_1.val)
        amplitude_2 = float(amplitude_slider_2.val)
        time_1 = float(time_1_textbox.text)
        time_4 = float(time_4_textbox.text)
        if time_1 != time_1__ or time_4 != time_4__:
            time_1__ = time_1
            time_4__ = time_4
            time_2 = initial_time_2
            time_3 = initial_time_3
            left_channel_1, left_channel_2, right_channel_1, right_channel_2 = generate_stereo_wave(f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_2, time_3, time_4)
            f1_, f2_, cf1_1_, cf2_1_, cf1_2_, cf2_2_, amplitude_1_, amplitude_2_, time_1_, time_4_ = f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_4
            graph(left_channel_1, left_channel_2, right_channel_1, right_channel_2, time_4)
            # グラフを更新
            fig.canvas.draw()
            combined_lift = left_channel_1 + left_channel_2
            combined_right = right_channel_1 + right_channel_2
            stereo_wave_1 = np.column_stack((combined_lift, combined_right))
            stereo_wave_1 = np.array(stereo_wave_1 * (2 ** 15 - 1), dtype=np.int16)
            print(0)
            print(time_1_)
            print(time_2)
            print(time_3)
        else:
            if f1 != f1_ or f2 != f2_ or cf1_1 != cf1_1_ or cf2_1 != cf2_1_ or cf1_2 != cf1_2_ or cf2_2 != cf2_2_ or amplitude_1 != amplitude_1_ or amplitude_2 != amplitude_2_:
                time_2 = initial_time_2
                time_3 = initial_time_3
                left_channel_1, left_channel_2, right_channel_1, right_channel_2 = generate_stereo_wave(f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_2, time_3, time_4)
                f1_, f2_, cf1_1_, cf2_1_, cf1_2_, cf2_2_, amplitude_1_, amplitude_2_, time_1_, time_4_ = f1, f2, cf1_1, cf2_1, cf1_2, cf2_2, amplitude_1, amplitude_2, time_1, time_4
                graph(left_channel_1, left_channel_2, right_channel_1, right_channel_2, time_4)
                # グラフを更新
                fig.canvas.draw()
                combined_audio_lift = left_channel_1 + left_channel_2
                combined_audio_right = right_channel_1 + right_channel_2
                stereo_wave_1 = np.column_stack((combined_audio_lift, combined_audio_right))
                combined_audio = np.array(stereo_wave_1 * (2 ** 15 - 1), dtype=np.int16)
                print(1)
                print(time_1_)
                print(time_2)
                print(time_3)
            else:
                print(2)
                print(time_1_)
                print(time_2)
                print(time_3)
                pass
        thread = threading.Thread(target=(play),args=(combined_audio,))
        thread.start()
        play_status_text.set_val("Playing")

# グラフとコントロールを表示するウィンドウを生成
fig = plt.figure(figsize=(15, 7.3), num = "前後変化うなりver4")
plt.subplots_adjust(left=0.12, bottom=0.44, top=0.95)

# グラフ表示エリアの設定
ax_left = plt.subplot(3, 1, 1)
ax_right = plt.subplot(3, 1, 2)
ax_beating = plt.subplot(3, 1, 3)

# 各スライダーは特定のパラメータを調整するためのGUIコントロールです
f1_slider_ax = plt.axes([0.12, 0.35, 0.78, 0.03])
f1_slider = Slider(f1_slider_ax, 'f_lift', 37, 47, valinit=initial_f1, valstep=2)

f2_slider_ax = plt.axes([0.12, 0.32, 0.78, 0.03])
f2_slider = Slider(f2_slider_ax, 'f_right', 37, 47, valinit=initial_f2, valstep=2)

f2_slider_ax.add_artist(f2_slider_ax.xaxis)
sl_xticks = np.arange(37, 49, 2)
f2_slider_ax.set_xticks(sl_xticks)

cf1_1_slider_ax = plt.axes([0.12, 0.25, 0.78, 0.03])
cf1_1_slider = Slider(cf1_1_slider_ax, 'cf_lift_1', 300, 1100, valinit=initial_cf1_1, valstep=0.5)

cf1_2_slider_ax = plt.axes([0.12, 0.22, 0.78, 0.03])
cf1_2_slider = Slider(cf1_2_slider_ax, 'cf_lift_2', 300, 1100, valinit=initial_cf1_2, valstep=0.5)

cf2_1_slider_ax = plt.axes([0.12, 0.19, 0.78, 0.03])
cf2_1_slider = Slider(cf2_1_slider_ax, 'cf_right_1', 300, 1100, valinit=initial_cf2_1, valstep=0.5)

cf2_2_slider_ax = plt.axes([0.12, 0.16, 0.78, 0.03])
cf2_2_slider = Slider(cf2_2_slider_ax, 'cf_right_2', 300, 1100, valinit=initial_cf2_2, valstep=0.5)

amplitude_slider_ax_1 = plt.axes([0.12, 0.13, 0.78, 0.03])
amplitude_slider_1 = Slider(amplitude_slider_ax_1, 'Amplitude_lift', 0, 1, valinit=initial_amplitude_1, valstep=0.1)

amplitude_slider_ax_2 = plt.axes([0.12, 0.10, 0.78, 0.03])
amplitude_slider_2 = Slider(amplitude_slider_ax_2, 'Amplitude_right', 0, 1, valinit=initial_amplitude_2, valstep=0.1)

# 確認ボタン1とテキストボックスの作成とイベントの登録
amplitude_textbox_ax_1 = plt.axes([0.15, 0.04, 0.08, 0.03])
amplitude_textbox_1 = TextBox(amplitude_textbox_ax_1, 'A_L:', initial=str(initial_amplitude_1))

amplitude_textbox_ax_2 = plt.axes([0.5, 0.04, 0.08, 0.03])
amplitude_textbox_2 = TextBox(amplitude_textbox_ax_2, 'A_R:', initial=str(initial_amplitude_2))

f1_textbox_ax = plt.axes([0.26, 0.04, 0.08, 0.03])
f1_textbox = TextBox(f1_textbox_ax, 'f_L:', initial=str(initial_f1))

f2_textbox_ax = plt.axes([0.61, 0.04, 0.08, 0.03])
f2_textbox = TextBox(f2_textbox_ax, 'f_R:', initial=str(initial_f2))

cf1_1_textbox_ax = plt.axes([0.38, 0.06, 0.08, 0.03])
cf1_1_textbox = TextBox(cf1_1_textbox_ax, 'cf_L_1:', initial=str(initial_cf1_1))

cf1_2_textbox_ax = plt.axes([0.38, 0.01, 0.08, 0.03])
cf1_2_textbox = TextBox(cf1_2_textbox_ax, 'cf_L_2:', initial=str(initial_cf1_2))

cf2_1_textbox_ax = plt.axes([0.73, 0.06, 0.08, 0.03])
cf2_1_textbox = TextBox(cf2_1_textbox_ax, 'cf_R_1:', initial=str(initial_cf2_1))

cf2_2_textbox_ax = plt.axes([0.73, 0.01, 0.08, 0.03])
cf2_2_textbox = TextBox(cf2_2_textbox_ax, 'cf_R_2:', initial=str(initial_cf2_2))

confirm_button_ax = plt.axes([0.82, 0.04, 0.08, 0.03])
confirm_button = Button(confirm_button_ax, 'Confirm', hovercolor='lightgray')

# スライダーとボタンの更新イベント1の登録
f1_slider.on_changed(update_beating_effect_1)
f2_slider.on_changed(update_beating_effect_1)
cf1_1_slider.on_changed(update_beating_effect_2)
cf2_1_slider.on_changed(update_beating_effect_2)
cf1_2_slider.on_changed(update_beating_effect_2)
cf2_2_slider.on_changed(update_beating_effect_2)
amplitude_slider_1.on_changed(update_beating_effect_2)
amplitude_slider_2.on_changed(update_beating_effect_2)
# 確認ボタン1のクリックイベントの登録
confirm_button.on_clicked(update_parameters)

# 時間の設定用テキストボックスと音声再生ボタンの作成
time_1_textbox_ax = plt.axes([0.04, 0.06, 0.08, 0.03])
time_1_textbox = TextBox(time_1_textbox_ax, 'time_1:', initial=str(initial_time_1))
time_4_textbox_ax = plt.axes([0.04, 0.01, 0.08, 0.03])
time_4_textbox = TextBox(time_4_textbox_ax, 'time_4:', initial=str(initial_time_4))
time_3_textbox_ax = plt.axes([0.91, 0.96, 0.08, 0.03])
time_3_textbox = TextBox(time_3_textbox_ax, 'time_1.5:')
time_3_textbox.set_active(False) 

#キャリア周波数設定用テキストボックス
cf_textbox_ax = plt.axes([0.02, 0.96, 0.08, 0.03])
cf_textbox = TextBox(cf_textbox_ax, 'cf:', initial=str(initial_cf))

# 音声再生ボタンの作成とクリック時のイベントハンドリング
save_button_ax1 = plt.axes([0.91, 0.06, 0.08, 0.03])
save_button1 = Button(save_button_ax1, 'audio play', hovercolor='lightgray')
save_button1.on_clicked(audio_play)

# 音声保存ボタンの作成とイベントハンドリング
save_button_ax = plt.axes([0.91, 0.01, 0.08, 0.03])
save_button = Button(save_button_ax, 'audio save', hovercolor='lightgray')
save_button.on_clicked(save_audio_file)

# 音声再生状態表示用テキストボックスの作成と初期化
play_status_text_ax = plt.axes([0.46, 0.96, 0.1, 0.03])
play_status_text = TextBox(play_status_text_ax, 'Play Status:', initial='Not Playing')
play_status_text.set_active(False) 
play_status_text_ax.set_facecolor('red')

# グラフの初期表示
left_channel_1, left_channel_2, right_channel_1, right_channel_2 = generate_stereo_wave(initial_f1, initial_f2, initial_cf1_1, initial_cf2_1, initial_cf1_2, initial_cf2_2, initial_amplitude_1, initial_amplitude_2, initial_time_1, initial_time_2, initial_time_3, initial_time_4)

graph(left_channel_1, left_channel_2, right_channel_1, right_channel_2, time_4)

plt.text(-3.4, 0, "* Amplitude_lift&right≤1", fontsize=15, color='red')

# ウィンドウを表示
plt.show()