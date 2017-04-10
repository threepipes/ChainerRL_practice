# -*- coding: utf-8 -*-
import pygame
import sys

"""
与えられた座標列を表示する
キーボードの左右で履歴の再生，巻き戻し
"""

def draw(pos_list):
    (w, h) = (400, 400)
    (x, y) = (w//2, h//2)
    pygame.init()
    pygame.display.set_mode((w, h), 0, 32)
    screen = pygame.display.get_surface()
    ox, oy = x, y
    turn = 0

    while True:
        pygame.display.update()
        pygame.time.wait(30)
        screen.fill((0, 20, 0, 0))
        if turn < len(pos_list):
            x = ox + int(pos_list[turn][0]*10)
            y = oy - int(pos_list[turn][1]*10)
        pygame.draw.circle(screen, (0, 200, 0), (x, y), 5)
        pygame.draw.line(screen, (100, 200, 0), (x, y), (ox, oy))
        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_LEFT] and turn > 0:
            turn -= 1
        if pressed[pygame.K_RIGHT] and turn < len(pos_list)-1:
            turn += 1

        for event in pygame.event.get():
            if is_exit(event):
                pygame.quit()
                return

def is_exit(event):
    return (
        event.type == pygame.KEYDOWN and
        event.key == pygame.K_ESCAPE or
        event.type == pygame.QUIT
    )
