import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
import pygame
from pymunk.pygame_util import DrawOptions

import torch
import torchvision.transforms as T
from PIL import Image


class BallOnBallEnv(gym.Env):
    """
    Gymnasium-Umgebung für das 2D-Physikspiel "Kugel auf Kugel".
    """

    # Erstelle die Kugeln
    BIG_RADIUS = 100
    BIG_MASS = 10

    SMALL_RADIUS = 30
    SMALL_MASS = 1

    # Parameter der Umgebung
    WIDTH = 800
    HEIGHT = 600
    GRAVITY = 1000
    OFFSETS = [0.1, -0.1]  # Random offsets
    force_amount = 700.0  # Kraft durch Aktionen

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode="rgb_array"):
        super(BallOnBallEnv, self).__init__()

        self.render_mode = render_mode


        self.ground_y = self.HEIGHT - 50

        # Aktionen: 0 = links, 1 = keine Aktion, 2 = rechts
        self.action_space = spaces.Discrete(3)

        # Zustandsraum: Position und Geschwindigkeit der kleinen Kugel
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([self.WIDTH, self.HEIGHT, np.inf, np.inf], dtype=np.float32),
        )

        # Pymunk Space erstellen
        self.space = pymunk.Space()
        self.space.gravity = (0, self.GRAVITY)
        self._create_static_line((0, self.ground_y), (self.WIDTH, self.ground_y))

        # Pygame Canvas und Optionen
        self.canvas = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.draw_options = DrawOptions(self.canvas)

        # Pygame-Fenster nur bei `human`
        self.screen = None
        self.clock = None

        # Kugeln erstellen
        self.big_body = None
        self.small_body = None
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

    def reset(self, seed=None, options=None):
        """Setzt die Umgebung zurück und gibt den Startzustand zurück."""
        super().reset(seed=seed)

        # Entferne alle dynamischen Bodies
        for body in self.space.bodies[:]:
            if body != self.space.static_body:
                for shape in body.shapes:
                    self.space.remove(shape)
                self.space.remove(body)


        big_pos = (self.WIDTH / 2, self.ground_y - self.BIG_RADIUS - 5)
        self.big_body = self._create_circle(self.BIG_MASS, self.BIG_RADIUS, big_pos, "gray")

        small_pos = (
            self.WIDTH / 2 + np.random.choice(self.OFFSETS),
            big_pos[1] - self.BIG_RADIUS - self.SMALL_RADIUS,
        )
        self.small_body = self._create_circle(self.SMALL_MASS, self.SMALL_RADIUS, small_pos, "black", friction=1)

        return self._get_state(), {}

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

        state = self._get_state()
        small_y = state[1]  # y-Position der kleinen Kugel
        big_y = self.big_body.position.y  # y-Position des Mittelpunkts der großen Kugel

        # Berechnung von done
        done = small_y >= (big_y - self.BIG_RADIUS * 0.75)

        reward = 1.0 if not done else -10.0

        if done:
            self.reset()

        if self.render_mode == "human":
            self.render()

        return state, reward, done, {}

    def render(self):
        """Zeichnet die Umgebung entweder auf den Bildschirm oder gibt das RGB-Array zurück."""
        # Hintergrund füllen
        self.canvas.fill((255, 255, 255))

        # Pymunk-Objekte auf die Canvas zeichnen
        self.space.debug_draw(self.draw_options)

        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
                self.clock = pygame.time.Clock()

            # Canvas auf das Fenster kopieren
            self.screen.blit(self.canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        return np.transpose(np.array(pygame.surfarray.array3d(self.canvas)), axes=(1, 0, 2))

    def close(self):
        """Schließt Pygame-Ressourcen."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None



def capture_screen(env):
    img_array = env.render()
    #img_array = np.transpose(img_array, (0, 1, 2))

    # Bildschirmhöhe und -breite
    screen_height, screen_width, _ = img_array.shape

    # Schneidet unwichtige Bereiche oben und unten ab
    cut_upper = 0.4  # 90% des Bildes behalten
    cut_lower = 0.8  # die unteren 10% abschneiden
    img_array = img_array[int(screen_height * cut_upper):int(screen_height * cut_lower), :, :]
    relevant_width = int(1 * screen_width)
    lower_ball_x = int(env.big_body.position.x)

    # Zentrieren um die Kugel
    if lower_ball_x < relevant_width // 2:
        slice_range = slice(0, relevant_width)
    elif lower_ball_x > (screen_width - relevant_width // 2):
        slice_range = slice(screen_width - relevant_width, screen_width)
    else:
        slice_range = slice(lower_ball_x - relevant_width // 2,
                            lower_ball_x + relevant_width // 2)

    centered_img_array = img_array[:, slice_range, :]
    norm_img_array = np.ascontiguousarray(centered_img_array, dtype=np.float32) / 255.0

    img_tensor = torch.from_numpy(norm_img_array).permute(2, 0, 1)
    pil_img_array = T.ToPILImage()(img_tensor)
    resized_img_array = T.Resize((60, 60), interpolation=Image.BICUBIC)(pil_img_array)
    grayscaled_img_array = T.Grayscale()(resized_img_array)
    final_tensor = T.ToTensor()(grayscaled_img_array)

    return final_tensor.unsqueeze(0)

# Beispiel-Test der Umgebung
if __name__ == "__main__":
    # Initialisiere die Umgebung
    env = BallOnBallEnv(render_mode="human")
    state = env.reset()
    print("Startzustand:", state)
    done = False

    # Initialisiere Pygame
    pygame.init()
    screen = pygame.display.set_mode((400, 300))  # Dummy-Fenster für Events
    env.render()
    res = capture_screen(env)


    while True:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
        print(env.render())
     #   screen_array = env.get_state()
        pass

    """print("Steuere mit den Pfeiltasten: Links (0), Nichts (1), Rechts (2).")

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
        pygame.quit()"""