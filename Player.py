#2023 ZHOU YUNHAO 

import os
import pygame
import re

# 定義音声フォルダのパス
audio_folder = os.getcwd() + r"\Save_file"

# 音声ファイルリストを取得し、数字の順に並べ替える
audio_files = sorted([file for file in os.listdir(audio_folder) if file.endswith('.wav')], key=lambda x: int(re.search(r'\d+', x).group()))
current_index = 0

# Pygameを初期化
pygame.init()

# ウィンドウを作成
window_width = 1000
window_height = 300
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('WAVプレイヤー')

# オーディオプレーヤーを作成
pygame.mixer.init()

def play_audio(index):
    if 0 <= index < len(audio_files):
        audio_file = os.path.join(audio_folder, audio_files[index])
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

def stop_audio():
    pygame.mixer.music.stop()

def show_centered_text(text, font_1, color):
    text_surface = font_1.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (window_width // 2, window_height // 2 - 75)
    window.blit(text_surface, text_rect)

# プレイ中かどうかを示すフラグ

font_1 = pygame.font.Font(None, 72)
font_2 = pygame.font.Font(None, 36)

playing = True

while playing:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            playing = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # マウスの左ボタンがクリックされたかどうかを確認
                x, y = event.pos
                if (window_width // 2 - 200) <= x <= (window_width // 2 - 50) and (window_height // 2 + 50) <= y <= (window_height // 2 + 100):  # 再生ボタン
                    play_audio(current_index)
                elif (window_width // 2 - 50) <= x <= (window_width // 2) and (window_height // 2 + 50) <= y <= (window_height // 2 + 100):  # 前のファイルボタン
                    current_index = (current_index - 1) % len(audio_files)
                    play_audio(current_index)
                elif (window_width // 2) <= x <= (window_width // 2 + 150) and (window_height // 2 + 50) <= y <= (window_height // 2 + 100):  # 一時停止ボタン
                    stop_audio()
                elif (window_width // 2 + 150) <= x <= (window_width // 2 + 200) and (window_height // 2 + 50) <= y <= (window_height // 2 + 100):  # 次のファイルボタン
                    current_index = (current_index + 1) % len(audio_files)
                    play_audio(current_index)
            if event.button == 3: # マウスの右ボタンがクリックされたかどうかを確認
                x, y = event.pos
                if (window_width // 2 - 200) <= x <= (window_width // 2 - 50) and (window_height // 2 + 50) <= y <= (window_height // 2 + 100):  # 再生ボタン
                    current_index = (current_index + 1) % len(audio_files)
                    # play_audio(current_index)

    window.fill((255, 255, 255))

    # ボタンを描画
    pygame.draw.rect(window, (0, 255, 0), (window_width // 2 - 200, window_height // 2 + 50, 150, 50))  # 再生ボタン
    pygame.draw.rect(window, (0, 0, 255), (window_width // 2 - 50, window_height // 2 + 50, 50, 50))  # 前のファイルボタン
    pygame.draw.rect(window, (255, 0, 0), (window_width // 2, window_height // 2 + 50, 150, 50))  # 一時停止ボタン
    pygame.draw.rect(window, (0, 0, 255), (window_width // 2 + 150, window_height // 2 + 50, 50, 50))  # 次のファイルボタン


    # ファイル名を描画
    if audio_files:
        current_file_name = audio_files[current_index]
        if current_index < 9:
            current_file_name = current_file_name[:1] + ":" + current_file_name[2:]
        else:
            current_file_name = current_file_name[:2] + ":" + current_file_name[3:]
        text = font_2.render(current_file_name, True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (window_width // 2, window_height // 2)
        window.blit(text, text_rect)

    # 音楽が終了したかどうかを確認し、終了した場合はメッセージを表示
    if not pygame.mixer.music.get_busy():
        show_centered_text('Ending', font_1, (255, 0, 0))
    else:
        show_centered_text('Playing', font_1, (0, 255, 0))

    text = font_2.render('start', True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (window_width // 2 - 125, window_height // 2 + 75)
    window.blit(text, text_rect)

    text = font_2.render('stop', True, (0, 0, 0))
    text_rect = text.get_rect()
    text_rect.center = (window_width // 2 + 75, window_height // 2 + 75)
    window.blit(text, text_rect)

    pygame.display.flip()

pygame.quit()