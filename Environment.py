import sys
import math
import time

import pygame
import pymunk
import pymunk.pygame_util

"""
Dieses Beispiel demonstriert ein 2D-Physik-Environment ähnlich dem CartPole-Setup.
Wir haben einen Boden (eine horizontale Linie), darauf eine große Kugel (Rad), 
auf der wiederum eine kleinere Kugel liegt. Durch die Pfeiltasten (Links/Rechts) 
kann auf die obere Kugel eine horizontale Kraft ausgeübt werden, wodurch diese 
die untere Kugel in Bewegung versetzt.

Wird die obere Kugel jedoch zu weit gedrückt oder fällt sie herunter, sodass sie 
den Boden berührt, wird das Environment zurückgesetzt.

Abhängigkeiten:
- pygame
- pymunk

Installation (wenn nötig):
    pip install pygame pymunk
"""

# Ein paar grundlegende Einstellungen
WIDTH, HEIGHT = 800, 600
FPS = 60
GRAVITY = 1000  # Pixel pro Sekunde^2


def create_static_line(space, start, end, thickness=5):
    """Erzeuge eine statische Linie im Raum."""
    body = space.static_body
    shape = pymunk.Segment(body, start, end, thickness)
    shape.friction = 1.0
    shape.elasticity = 0.0
    space.add(shape)
    return shape


def create_circle(space, mass, radius, pos, friction=1.0, elasticity=0.0):
    """Erzeuge einen Kreis mit gegebener Masse, Radius und Startposition."""
    moment = pymunk.moment_for_circle(mass, 0, radius)
    body = pymunk.Body(mass, moment)
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.friction = friction
    shape.elasticity = elasticity
    space.add(body, shape)
    return body, shape


def reset_environment(space, ground_y):
    """Setzt die Positionen der Kugeln zurück."""
    # Entferne alle dynamischen Bodies (bis auf static Bodies) aus dem Space
    for shape in space.shapes[:]:
        if not isinstance(shape, pymunk.Segment):
            space.remove(shape)
    for body in space.bodies[:]:
        if body != space.static_body:
            space.remove(body)

    # Erzeuge die Kugeln neu
    big_radius = 50
    big_mass = 10
    big_pos = (WIDTH / 2, ground_y - big_radius - 1)
    big_body, big_shape = create_circle(space, big_mass, big_radius, big_pos, friction=1.0, elasticity=0.0)

    small_radius = 20
    small_mass = 1
    small_pos = (WIDTH / 2, big_body.position.y - big_radius - small_radius - 1)
    small_body, small_shape = create_circle(space, small_mass, small_radius, small_pos, friction=1.0, elasticity=0.0)

    return big_body, small_body


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Physics Environment - Kugel auf Kugel")
    clock = pygame.time.Clock()

    # Pymunk Space erstellen
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    # Bodenlinie erstellen
    ground_y = HEIGHT - 50
    ground = create_static_line(space, (0, ground_y), (WIDTH, ground_y))

    # Erzeuge anfängliche Kugeln
    big_body, small_body = reset_environment(space, ground_y)

    # Steuerungs-Parameter
    force_amount = 500.0  # Kraft die auf die kleine Kugel ausgeübt wird

    running = True
    while running:
        dt = 1.0 / FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        # Kraft auf die kleine Kugel anwenden, wenn Pfeiltasten gedrückt sind
        if keys[pygame.K_LEFT]:
            # Kraft nach links
            small_body.apply_force_at_local_point((-force_amount, 0), (0, 0))
        if keys[pygame.K_RIGHT]:
            # Kraft nach rechts
            small_body.apply_force_at_local_point((force_amount, 0), (0, 0))

        # Physik updaten
        space.step(dt)

        # Prüfen, ob die obere Kugel auf den Boden gefallen ist:
        # Wenn die Unterkante der oberen Kugel unter/hinter dem Boden liegt, ist sie gefallen
        if small_body.position.y + 20 >= ground_y:
            # Environment zurücksetzen
            big_body, small_body = reset_environment(space, ground_y)

        # Bildschirm aktualisieren
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)

        # Zusätzliche Hilfslinien oder Texte:
        font = pygame.font.SysFont("Arial", 20)
        text = font.render("Links/Rechts Pfeiltasten: obere Kugel horizontal bewegen. Fällt sie zu Boden, Reset.", True,
                           (0, 0, 0))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()
