import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pygame
from pymunk.pygame_util import DrawOptions


class CustomDrawOptions(DrawOptions):
    def draw_circle(self, pos, radius, angle, outline_color, fill_color):
        # Farben je nach Ballgröße definieren
        if radius > 30:  # Großer Ball
            fill_color = (0, 255, 0)  # Grün
        else:  # Kleiner Ball
            fill_color = (0, 0, 0)  # Schwarz

        # Zeichne den Kreis ohne Querstrich
        pygame.draw.circle(self.surface, fill_color, (int(pos[0]), int(pos[1])), int(radius))

class BallOnBallEnv(gym.Env):
    """
    Gymnasium-Umgebung für das 2D-Physikspiel "Kugel auf Kugel".
    """

    def __init__(self):
        super(BallOnBallEnv, self).__init__()

        # Parameter der Umgebung
        self.WIDTH = 800
        self.HEIGHT = 600
        self.GRAVITY = 1000
        self.force_amount = 700.0  # Kraft durch Aktionen
        self.ground_y = self.HEIGHT - 50

        # Aktionen: [-1] für links, [0] für keine Aktion, [1] für rechts
        self.action_space = spaces.Discrete(3)

        # Zustandsraum: Position und Geschwindigkeit der kleinen Kugel
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf]),
            high=np.array([self.WIDTH, self.HEIGHT, np.inf, np.inf]),
            dtype=np.float32
        )

        # Pymunk Space erstellen
        self.space = pymunk.Space()
        self.space.gravity = (0, self.GRAVITY)
        self._create_static_line((0, self.ground_y), (self.WIDTH, self.ground_y))

        # Kugeln erstellen
        self.big_body = None
        self.small_body = None
        self.screen = None
        self.clock = None
        self.draw_options = CustomDrawOptions(self.screen)
        self.reset()

    def _create_static_line(self, start, end, thickness=5):
        body = self.space.static_body
        shape = pymunk.Segment(body, start, end, thickness)
        shape.friction = 1.0
        shape.elasticity = 0.0
        self.space.add(shape)

    def _create_circle(self, mass, radius, pos, color, friction=0):
        moment = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, moment)
        body.position = pos
        shape = pymunk.Circle(body, radius)
        shape.friction = friction
        shape.elasticity = 0.0
        shape.color = pygame.Color(color)

        self.space.add(body, shape)
        return body

    def reset(self):
        """Setzt die Umgebung zurück und gibt den Startzustand zurück."""
        # Entferne alle dynamischen Bodies
        for body in self.space.bodies[:]:
            if body != self.space.static_body:
                for shape in body.shapes:
                    self.space.remove(shape)
                self.space.remove(body)

        # Erstelle die Kugeln
        big_radius = 50
        big_mass = 10
        big_pos = (self.WIDTH / 2, self.ground_y - big_radius - 1)
        self.big_body = self._create_circle(big_mass, big_radius, big_pos, "red")

        small_radius = 20
        small_mass = 1
        small_pos = (self.WIDTH / 2, big_pos[1] - big_radius - small_radius - 1)
        self.small_body = self._create_circle(small_mass, small_radius, small_pos, "black", friction=1)

        # Anfangszustand
        return self._get_state()

    def _get_state(self):
        """Gibt den aktuellen Zustand der Umgebung zurück."""
        pos = self.small_body.position
        vel = self.small_body.velocity
        return np.array([pos.x, pos.y, vel.x, vel.y], dtype=np.float32)

    def step(self, action):
        """Führt eine Aktion aus und gibt die Ergebnisse zurück."""
        if action == 0:  # Links
            self.small_body.apply_force_at_local_point((-self.force_amount, 0))

        elif action == 2:  # Rechts
            self.small_body.apply_force_at_local_point((self.force_amount, 0))

        # Physik aktualisieren
        self.space.step(1 / 60.0)

        # Zustand, Belohnung und Fertigstellungsstatus
        state = self._get_state()
        done = state[1] + 25 > self.ground_y  # Wenn die kleine Kugel auf den Boden fällt
        reward = 1.0 if not done else -10.0

        if done:
            self.reset()

        return state, reward, done, {}

    def render(self, mode="human"):
        """Optional: Rendering der Umgebung."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()
            self.draw_options = DrawOptions(self.screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)

        # Aktualisieren des Fensters
        pygame.display.update()
        self.clock.tick(60)

    def close(self):
        """Optionale Aufräumarbeiten."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.draw_options = None

# Beispiel-Test der Umgebung
if __name__ == "__main__":
    # Initialisiere die Umgebung
    env = BallOnBallEnv()
    state = env.reset()
    print("Startzustand:", state)
    done = False

    # Initialisiere Pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 300))  # Dummy-Fenster für Events

    print("Steuere mit den Pfeiltasten: Links (0), Nichts (1), Rechts (2).")

    try:
        while not done:
            # Standardaktion: Nichts (1)
            action = 1

            # Prüfe, ob eine Taste gedrückt ist
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 0  # Aktion: Links
            elif keys[pygame.K_RIGHT]:
                action = 2  # Aktion: Rechts
            elif keys[pygame.K_UP]:
                action = 1  # Aktion: Nichts

            # Überprüfe andere Events, wie das Schließen des Fensters
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            # Schritt in der Umgebung
            state, reward, done, info = env.step(action)
            env.render()
            print(f"State: {state}, Reward: {reward}, Done: {done}")
    finally:
        env.close()
        pygame.quit()