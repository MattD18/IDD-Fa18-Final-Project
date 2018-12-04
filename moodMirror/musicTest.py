import pygame


import time


pygame.mixer.init()
happy = pygame.mixer.Sound('happy.wav')
sad = pygame.mixer.Sound('sad.wav')



counter = 0
happy.play()
while True:
    time.sleep(5)
    counter += 1
    if counter % 2 == 1:
        pygame.mixer.stop()
        sad.play()
    else:
        pygame.mixer.stop()
        happy.play()
