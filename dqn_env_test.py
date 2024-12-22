import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import pygame
import gymnasium as gym
import pymunk

from pymunk.pygame_util import DrawOptions

# =========================
# 0) Gerät für Torch auswählen (mps -> cuda -> cpu)
# =========================
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"[INFO] Verwende Gerät: {device}")


# =========================
# Environment mit abschaltbarem Rendering
# und höherer Auflösung
# =========================
class CustomDrawOptions(DrawOptions):
    """ Nur für farbige Kreise (Großer = Grün, Kleiner = Schwarz). """
    def draw_circle(self, pos, radius, angle, outline_color, fill_color):
        if radius > 30:  # Großer Ball
            fill_color = (0, 255, 0)
        else:  # Kleiner Ball
            fill_color = (0, 0, 0)
        pygame.draw.circle(self.surface, fill_color, (int(pos[0]), int(pos[1])), int(radius))


class BallOnBallEnv(gym.Env):
    """
    Gymnasium-Umgebung für das 2D-Physikspiel "Kugel auf Kugel".
    """
    metadata = {"render_modes": ["human", "none"]}

    def __init__(self, render_mode="none"):
        """
        :param render_mode: "none" => kein Rendering, "human" => Pygame-Fenster öffnen.
        """
        super().__init__()
        self.render_mode = render_mode

        # Parameter der Umgebung
        self.WIDTH = 800
        self.HEIGHT = 600
        self.GRAVITY = 1000
        self.OFFSETS = [0.1, -0.1]
        self.force_amount = 700.0
        self.ground_y = self.HEIGHT - 50

        # Aktionen: [-1] für links, [0] für keine Aktion, [1] für rechts
        self.action_space = gym.spaces.Discrete(3)

        # Beobachtungen: pos.x, pos.y, vel.x, vel.y
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -np.inf, -np.inf]),
            high=np.array([self.WIDTH, self.HEIGHT, np.inf, np.inf]),
            dtype=np.float32
        )

        # Pymunk Space
        self.space = pymunk.Space()
        self.space.gravity = (0, self.GRAVITY)
        self._create_static_line((0, self.ground_y), (self.WIDTH, self.ground_y))

        # Kugeln / Bodies
        self.big_body = None
        self.small_body = None
        self.screen = None
        self.clock = None
        self.draw_options = None

        self.reset()  # Erstes Setup

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
        """ Setzt die Umgebung zurück und gibt den Startzustand zurück. """
        # Entferne alte Bodies
        for body in self.space.bodies[:]:
            if body != self.space.static_body:
                for shape in body.shapes:
                    self.space.remove(shape)
                self.space.remove(body)

        # Erstelle die Kugeln
        big_radius = 150
        big_mass = 10
        big_pos = (self.WIDTH / 2, self.ground_y - big_radius - 1)
        self.big_body = self._create_circle(big_mass, big_radius, big_pos, "gray")

        small_radius = 60
        small_mass = 1
        small_pos = (
            self.WIDTH / 2 + np.random.choice(self.OFFSETS),
            big_pos[1] - big_radius - small_radius - 1
        )
        self.small_body = self._create_circle(small_mass, small_radius, small_pos, "black", friction=1)

        if self.render_mode == "human":
            self._init_render()

        return self._get_state()

    def _init_render(self):
        """Nur initiales Setup für das Pygame-Fenster, wenn render_mode='human'."""
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            self.clock = pygame.time.Clock()
            self.draw_options = CustomDrawOptions(self.screen)

    def _get_state(self):
        """Zustand: [pos_x, pos_y, vel_x, vel_y]."""
        pos = self.small_body.position
        vel = self.small_body.velocity
        return np.array([pos.x, pos.y, vel.x, vel.y], dtype=np.float32)

    def step(self, action):
        """Führt die Aktion aus und gibt (state, reward, done, info) zurück."""
        if action == 0:  # Links
            self.small_body.apply_force_at_local_point((-self.force_amount, 0))
        elif action == 2:  # Rechts
            self.small_body.apply_force_at_local_point((self.force_amount, 0))
        # (action==1 => keine Aktion)

        self.space.step(1 / 60.0)

        state = self._get_state()
        done = state[1] + 80 > self.ground_y
        reward = 1.0 if not done else -10.0

        if done:
            self.reset()

        return state, reward, done, {}

    def render(self):
        """Zeichnet die Umgebung, falls render_mode='human'."""
        if self.render_mode != "human":
            return  # Nichts tun, wenn kein Rendering gewünscht

        if self.screen is None:
            self._init_render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.update()
        self.clock.tick(60)

    def get_state(self):
        """
        Liefert ein gerendertes Bild als Graustufenarray mit Auflösung (80x60),
        transponiert und ggf. gecroppt -> z.B. (52, 80).
        """
        if self.render_mode != "human":
            # Auch wenn wir nicht "rendern", können wir ein verstecktes Surface nutzen:
            if self.screen is None:
                # Offscreen-Rendering:
                pygame.init()
                self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
                self.draw_options = CustomDrawOptions(self.screen)
            # Simuliertes "Rendern" ins Surface
            self.screen.fill((255, 255, 255))
            self.space.debug_draw(self.draw_options)

        if self.screen is None:
            return None

        # Skaliere auf 80x60
        scaled = pygame.transform.scale(self.screen, (80, 60))
        arr = pygame.surfarray.array3d(scaled)

        # In Graustufen umwandeln
        final_gray = (0.299 * arr[:, :, 0] +
                      0.587 * arr[:, :, 1] +
                      0.114 * arr[:, :, 2])

        # (60,80) => Transpose => evtl. Crop
        gray_t = final_gray.astype(np.uint8).T
        # z.B. Crop 6:-2 => (52,80)
        cropped = gray_t[6:-2] / 255.0

        return torch.tensor(cropped, dtype=torch.float32)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.draw_options = None


# =========================
# 1) Einfaches CNN-Modell mit dynamischer Flatten-Bestimmung
# =========================
class DQNCNN(nn.Module):
    def __init__(self, num_stacked=4, num_actions=3):
        super(DQNCNN, self).__init__()
        # num_stacked => Anzahl der gestapelten Differenzbilder als Kanäle

        self.conv = nn.Sequential(
            nn.Conv2d(num_stacked, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU()
        )
        # Wir definieren self.fc erst dynamisch
        self.fc = None
        self.fc_init_done = False
        self.num_actions = num_actions

    def forward(self, x):
        """
        x: [Batch, num_stacked, H, W]
        """
        x = self.conv(x)

        # Dynamische Bestimmung der Flatten-Dimension
        if not self.fc_init_done:
            flat_size = x.size(1) * x.size(2) * x.size(3)
            self.fc = nn.Sequential(
                nn.Linear(flat_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.num_actions)
            )
            self.fc_init_done = True
            self.fc.to(x.device)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# =========================
# 2) Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        state, next_state : Torch-Tensor [num_stacked, H, W]
        """
        self.buffer.append((
            state.to(dtype=torch.float32),  # float32 sicherstellen
            action,
            reward,
            next_state.to(dtype=torch.float32),  # float32 sicherstellen
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # states: tuple von [num_stacked, H, W], wir stapeln -> [batch_size, num_stacked, H, W]
        return (torch.stack(states),
                torch.tensor(actions, dtype=torch.long, device=device),
                torch.tensor(rewards, dtype=torch.float32, device=device),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32, device=device))

    def __len__(self):
        return len(self.buffer)


# =========================
# 3) DQN-Agent
# =========================
class DQNAgent:
    def __init__(
            self,
            num_actions=3,
            gamma=0.99,
            lr=1e-4,
            batch_size=32,
            eps_start=1.0,
            eps_end=0.01,
            eps_decay=10000,
            target_update_interval=1000,
            replay_capacity=50000,
            num_stacked=4
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_interval = target_update_interval
        self.num_stacked = num_stacked

        # Q-Netzwerk + Target-Netzwerk
        self.policy_net = DQNCNN(num_stacked=num_stacked, num_actions=num_actions).to(device)
        self.target_net = DQNCNN(num_stacked=num_stacked, num_actions=num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self.steps_done = 0

    @property
    def epsilon(self):
        """Aktueller Epsilon-Wert für Epsilon-Greedy."""
        return self.eps_end + (self.eps_start - self.eps_end) * \
               np.exp(-1.0 * self.steps_done / self.eps_decay)

    def select_action(self, state_stacked):
        """
        state_stacked: [num_stacked, H, W], Torch-Tensor
        => Epsilon-Greedy-Auswahl
        """
        # Falls Batch-Dimension fehlt: [1, num_stacked, H, W]
        if state_stacked.ndimension() == 3:
            state_stacked = state_stacked.unsqueeze(0)

        eps_threshold = self.epsilon
        self.steps_done += 1

        if random.random() < eps_threshold:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_stacked.to(device))
                return q_values.argmax(dim=1).item()

    def optimize(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Konvertiere sicher alle Tensoren auf float32 und device
        states = states.to(device, dtype=torch.float32)
        next_states = next_states.to(device, dtype=torch.float32)
        actions = actions.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        # Q(s,a) aus dem policy network
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Max a' Q_target(s', a')
        with torch.no_grad():
            q_next = self.target_net(next_states).max(dim=1)[0]
            q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = nn.MSELoss()(q_values, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Regelmäßiges Update des Target-Netzwerks
        if self.steps_done % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss

# =========================
# 4) Haupttraining
# =========================
def train_dqn(env, num_episodes=500, num_stacked=4):
    """
    Trainiert DQN mit Frame-Stacking mehrerer Differenzbilder.
    """
    agent = DQNAgent(num_actions=env.action_space.n, num_stacked=num_stacked)

    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        env.reset()

        # Wir sammeln "num_stacked" Frames, wobei jeder Frame ein Differenzbild sein wird.
        # Für die allerersten Frames nehmen wir 0 - 0 = 0, also leere Differenz.
        # => shape (H, W)
        current_frame = env.get_state()
        if current_frame is None:
            # Wenn kein Bild zurückkommt, brechen wir ab
            continue

        current_frame = current_frame.to(device)
        # Stapel an Differenzbildern => deque maxlen=num_stacked
        diff_stack = deque(maxlen=num_stacked)

        # Initial befüllen => die ersten "num_stacked" Differenzbilder = 0
        for _ in range(num_stacked):
            diff_stack.append(torch.zeros_like(current_frame))

        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0

        while not done:
            # State = Stapel aller Differenzbilder => shape (num_stacked, H, W)
            state_stacked = torch.stack(list(diff_stack), dim=0)

            # Aktion wählen
            action = agent.select_action(state_stacked)

            # Schritt in der Umgebung
            _, reward, done, _ = env.step(action)
            episode_reward += reward

            # Nächster Frame
            next_frame = env.get_state()
            if next_frame is None:
                done = True
                break
            next_frame = next_frame.to(device)

            # Differenzbild: next_frame - current_frame
            diff_img = next_frame - current_frame

            # Replay: (state_stacked, action, reward, next_state_stacked, done)
            # => Nächster Stapel: diff_stack + diff_img
            new_stack = deque(diff_stack, maxlen=num_stacked)  # Kopie
            new_stack.append(diff_img)
            next_state_stacked = torch.stack(list(new_stack), dim=0)

            agent.replay_buffer.push(
                state_stacked.cpu(),
                action,
                reward,
                next_state_stacked.cpu(),
                float(done)
            )

            # Optimize
            loss_val = agent.optimize()
            if loss_val is not None:
                episode_loss += loss_val.item()
                loss_count += 1

            # Vorbereiten für nächste Schleife
            diff_stack = new_stack
            current_frame = next_frame

        mean_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        episode_rewards.append(episode_reward)
        episode_losses.append(mean_loss)

        print(
            f"Episode {episode+1}/{num_episodes}, "
            f"Return: {episode_reward:.2f}, "
            f"Loss: {mean_loss:.4f}, "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    print("[INFO] Training abgeschlossen.")
    return agent, episode_rewards, episode_losses


# =========================
# 5) Evaluierungsfunktion
# =========================
def evaluate_agent(env, agent, episodes=5, num_stacked=4):
    """
    Evaluierung mit minimaler Exploration (agent.eps_start=0.01) und Rendern (falls gewünscht).
    """
    old_eps_start = agent.eps_start
    agent.eps_start = 0.01  # Minimale Exploration

    total_reward = 0.0

    for ep in range(episodes):
        env.reset()
        current_frame = env.get_state()
        if current_frame is None:
            break
        current_frame = current_frame.to(device)

        # Anfang: Stapel von Nullbildern
        diff_stack = deque(maxlen=num_stacked)
        for _ in range(num_stacked):
            diff_stack.append(torch.zeros_like(current_frame))

        done = False
        ep_reward = 0.0

        while not done:
            env.render()  # Nur wenn render_mode='human'

            state_stacked = torch.stack(list(diff_stack), dim=0)
            action = agent.select_action(state_stacked)

            _, reward, done, _ = env.step(action)
            ep_reward += reward

            next_frame = env.get_state()
            if next_frame is None:
                done = True
                break
            next_frame = next_frame.to(device)

            diff_img = next_frame - current_frame

            diff_stack.append(diff_img)
            current_frame = next_frame

        print(f"[EVAL] Episode {ep+1}, Reward: {ep_reward:.2f}")
        total_reward += ep_reward

    avg_reward = total_reward / episodes
    print(f"[EVAL] Durchschnittlicher Reward: {avg_reward:.2f}")

    agent.eps_start = old_eps_start


# =========================
# 6) Hauptprogramm
# =========================
if __name__ == "__main__":
    # Hier schalten wir das Rendering ab => render_mode="none"
    # Für Training willst du i.d.R. KEIN Rendering, um Performance zu sparen.
    env = BallOnBallEnv(render_mode="none")

    agent, rewards, losses = train_dqn(env, num_episodes=200, num_stacked=4)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward pro Episode")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.title("Loss pro Episode")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluation mit eingeschaltetem Rendering (falls man sehen will)
    # => Du kannst das in einer separaten Env machen oder:
    env.close()
    eval_env = BallOnBallEnv(render_mode="human")
    evaluate_agent(eval_env, agent, episodes=5, num_stacked=4)
    eval_env.close()