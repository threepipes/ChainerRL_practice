# -*- coding: utf-8 -*-
import pygame
import sys

def draw(pos_list):
    (w,h) = (1024,1024)   # 画面サイズ
    (x,y) = (w//2, h//2)
    pygame.init()       # pygame初期化
    pygame.display.set_mode((w, h), 0, 32)  # 画面設定
    screen = pygame.display.get_surface()
    ox, oy = x, y
    turn = 0

    while True:
        pygame.display.update()     # 画面更新
        pygame.time.wait(30)        # 更新時間間隔
        screen.fill((0, 20, 0, 0))
        if turn < len(pos_list):
            x = ox + int(pos_list[turn][0]*10)
            y = oy - int(pos_list[turn][1]*10)
        # 円を描画
        pygame.draw.circle(screen, (0, 200, 0), (x, y), 5)
        pygame.draw.line(screen, (100, 200, 0), (x, y), (ox, oy))
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_LEFT] and turn > 0:
            turn -= 1
        if pressed[pygame.K_RIGHT] and turn < len(pos_list)-1:
            turn += 1

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                # ESCキーなら終了
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
