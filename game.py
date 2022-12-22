from dataclasses import dataclass, field
import numpy as np
import pygame
import scipy.spatial
from typing import Any

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def print_return(f):
    def call(*args, **kwargs):
        out = f(*args, **kwargs)
        print(f'`{f.__name__}` returned\n{out}')
        return out
    return call

@print_return
def polygon(n, radius):
    theta = np.linspace(0, 2 * np.pi - 2 * np.pi / n, n)
    return radius * np.stack([
        np.sin(theta),
        np.cos(theta),
        np.zeros_like(theta),
    ]).T

@dataclass
class Game:
    # Settings
    screen_size: np.ndarray = field(default_factory=lambda: np.array([500, 500], dtype=int))
    color_background: tuple[int, int, int] = WHITE
    color_foreground: tuple[int, int, int] = BLACK

    # State
    running: bool = True
    keys: set[int] = field(default_factory=set)
    angles: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    r: np.ndarray = field(default_factory=lambda: polygon(6, 30))
    r0: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    # Resources
    screen: Any = None

    @property
    def origin(self):
        return np.array([*(self.screen_size / 2), 0], dtype=float)
    
    @property
    def R(self):
        return (scipy.spatial
            .transform
            .Rotation
            .from_euler('ZXY', self.angles * np.pi / 180)
            .as_matrix())
    
    def process_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.KEYDOWN:
            self.keys.add(event.key)
        elif event.type == pygame.KEYUP:
            if event.key in self.keys:
                if event.key == pygame.K_q:
                    self.angles[0] += 15
                elif event.key == pygame.K_e:
                    self.angles[0] -= 15
                elif event.key == pygame.K_w:
                    self.angles[1] += 15
                elif event.key == pygame.K_s:
                    self.angles[1] -= 15
                elif event.key == pygame.K_a:
                    self.angles[2] -= 15
                elif event.key == pygame.K_d:
                    self.angles[2] += 15
                elif event.key == pygame.K_UP:
                    self.r0[1] -= 50
                elif event.key == pygame.K_DOWN:
                    self.r0[1] += 50
                elif event.key == pygame.K_LEFT:
                    self.r0[0] -= 50
                elif event.key == pygame.K_RIGHT:
                    self.r0[0] += 50
            self.keys.remove(event.key)
    
    def process_events(self):
        for event in pygame.event.get():
            self.process_event(event)
    
    def loop(self):
        # Process events
        self.process_events()
        
        # Compute screen coordinates
        offset = self.r0 + self.origin
        offset = offset[np.newaxis, :]
        r = np.zeros_like(self.r)
        for i in range(len(r)):
            r[i, :] = self.R @ self.r[i, :]
        r = (r + offset)[:, : 2]
        
        # Draw commands
        self.screen.fill(self.color_background)
        pygame.draw.polygon(
            surface=self.screen,
            color=self.color_foreground,
            points=r,
        )
        pygame.display.flip()
    
    def run(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.screen_size)
        self.running = True
        while self.running:
            self.loop()
        pygame.quit()

Game().run()
