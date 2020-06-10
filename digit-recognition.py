import pygame
from pygame.locals import *
import torch 
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import sys
from network import DigitNet
import time

pygame.init()
pygame.font.init()

width = 500
height = 500
erase_time_s = 1
padding_factor = 0.2

window = pygame.display.set_mode((width, height))
pygame.display.set_caption('Digit Recognition')
mouse = pygame.mouse
canvas = window.copy()
font = pygame.font.SysFont('Arial', 30)

BLACK = pygame.Color(0, 0 ,0)
WHITE = pygame.Color(255, 255, 255)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
model = DigitNet().to(device)
model.load_state_dict(torch.load('model.ckpt'))

start = False
time_elapsed_s = 0
while True:
    time_start = time.time()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
      
    window.fill(BLACK)

    if mouse.get_pressed()[0]:
        pygame.draw.circle(canvas, WHITE, pygame.mouse.get_pos(), 10)
        start = True
        time_elapsed_s = 0
         
    window.blit(canvas, (0, 0))
    
    if start:
        image = pygame.image.tostring(window,"RGB",False)
        image = Image.frombytes("RGB",(500,500), image).convert('1')
        image = image.crop(image.getbbox())
        width, height = image.size
        if width > height:
            padding = int(width*padding_factor)
            image = ImageOps.expand(image, (padding, (width-height)//2+padding, padding, 
                                            width-height-(width-height)//2+padding))
        elif height > width:
            padding = int(height*padding_factor)
            image = ImageOps.expand(image, ((height-width)//2+padding, padding, 
                                            height-width-(height-width)//2+padding, padding))
        
        transform = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor()])
        
        image.save('digit.png')
        image = transform(image).to(device)
        image = image.unsqueeze(0)
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        label = predicted[0].cpu().numpy()

        window.blit(font.render(str(label), False, WHITE), (470,450))
    else:
        window.blit(font.render("?", False, WHITE), (470,450))
    
    time_elapsed_s += time.time()-time_start    
    
    if time_elapsed_s >= erase_time_s:
        canvas.fill(BLACK)
        start = False
        time_elapsed_s = 0
    
    pygame.display.update()
    