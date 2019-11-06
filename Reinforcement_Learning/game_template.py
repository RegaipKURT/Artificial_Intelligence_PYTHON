import pygame
import random

WIDTH = 360
HEIGHT = 360
FPS = 30
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DQL OYUNU")
clock = pygame.time.Clock()

#colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
GREEN = (0,255,0)

running = True
while running:
    
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    #update element

    #draw and render
    screen.fill(GREEN)
    pygame.display.flip()

pygame.quit()