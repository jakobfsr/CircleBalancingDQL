import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


# Gerät für Torch auswählen
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
print(f"Verwendetes Gerät: {device}")

# =========================
# 1) Einfaches CNN-Modell
# =========================
class DQNCNN(nn.Module):
    def __init__(self, num_actions=3):
        super(DQNCNN, self).__init__()
        # Wir nehmen an, dass das Eingabebild ca. Höhe=22, Breite=40 hat.

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2),  # -> (16, 10, 19) ungefähr
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # -> (32, 4, 8) ungefähr
            nn.ReLU()
        )
        # Aus Flatten kommt etwa 32*4*8 = 1024 Neuronen (bei exakt 22x40 Eingabe).
        # self.fc = nn.Sequential(
        #     nn.Linear(32 * 4 * 8, 128),  # alt
        #     ...
        # )

        self.fc = nn.Sequential(
            nn.Linear(32 * 11 * 18, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        # Debug: print(f"Shape after convolution: {x.shape}")
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
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states),                # shape: [batch_size, 1, H, W]
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(next_states),           # shape: [batch_size, 1, H, W]
                torch.tensor(dones, dtype=torch.float32))

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
            replay_capacity=50000
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_interval = target_update_interval

        # Q-Netzwerk + Target-Netzwerk
        self.policy_net = DQNCNN(num_actions)
        self.target_net = DQNCNN(num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self.steps_done = 0

    def select_action(self, state):
        """
        Epsilon-Greedy: Mit Wahrscheinlichkeit eps -> random Action,
        sonst beste Action über Q-Netzwerk.

        state: torch.Tensor [1, 1, H, W]
        """
        # Stelle sicher, dass wir einen Batch-Dim haben:
        if state.ndimension() == 3:  # (Channels, H, W)
            state = state.unsqueeze(0)  # -> (1, Channels, H, W)

        # Epsilon-Berechnung:
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        np.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if random.random() < self.eps_threshold:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state.unsqueeze(0).unsqueeze(0))
                return q_values.argmax(dim=1).item()

    def optimize(self):
        """
        Ein einzelner Trainingsschritt mit einer Batch aus dem Replay-Speicher.
        Gibt den Loss zurück, um ihn im Training zu protokollieren.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None  # Noch zu wenige Übergänge, kein Training

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Falls GPU gewünscht, hier .to(device) ergänzen
        # states, next_states = states.to(device), next_states.to(device)
        # actions, rewards, dones = actions.to(device), rewards.to(device), dones.to(device)

        # Q(s,a) aus dem policy network
        q_values = self.policy_net(states)
        # gather => wähle die Q-Werte an den ausgeführten Actions
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
def train_dqn(env, num_episodes=500):
    agent = DQNAgent(num_actions=env.action_space.n)

    episode_rewards = []
    episode_losses = []

    for episode in range(num_episodes):
        env.reset()
        env.render()

        # Frame 1:
        current_frame = env.get_state()
        if current_frame is None:
            # Falls das Environment doch kein Bild zurückgibt, skip
            continue
        if not torch.is_tensor(current_frame):
            current_frame = torch.tensor(current_frame, dtype=torch.float32)

        # Zu Beginn haben wir kein prev_frame, also diff = 0
        prev_frame = current_frame.clone()
        state_img = torch.zeros_like(current_frame)  # Erster Zustand => Null-Differenz

        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0

        while not done:
            # Aktion auswählen basierend auf dem Differenzbild
            action = agent.select_action(state_img)

            # Schritt in der Umgebung
            _, reward, done, _ = env.step(action)
            episode_reward += reward

            # Nächsten Frame holen
            next_frame = env.get_state()
            if next_frame is None:
                # Falls kein Bild -> Break
                done = True
                break
            if not torch.is_tensor(next_frame):
                next_frame = torch.tensor(next_frame, dtype=torch.float32)

            # Differenz berechnen (Movement)
            next_state_img = next_frame - current_frame

            # Replay speichern
            agent.replay_buffer.push(
                state_img.unsqueeze(0),  # shape (1, H, W)
                action,
                reward,
                next_state_img.unsqueeze(0),
                done
            )

            # Training
            loss_val = agent.optimize()
            if loss_val is not None:
                episode_loss += loss_val.item()
                loss_count += 1

            # Vorbereiten für nächste Schleife
            prev_frame = current_frame
            current_frame = next_frame
            state_img = next_state_img

        mean_loss = episode_loss / loss_count if loss_count > 0 else 0.0
        episode_rewards.append(episode_reward)
        episode_losses.append(mean_loss)

        print(f"Episode {episode + 1}/{num_episodes}, Return: {episode_reward:.2f}, Loss: {mean_loss:.4f}, Epsilon: {agent.eps_threshold}")

    print("Training abgeschlossen.")
    return agent, episode_rewards, episode_losses


# =========================
# 5) Evaluierungsfunktion
# =========================
def evaluate_agent(env, agent, episodes=5):
    """
    Lässt den Agenten für ein paar Episoden laufen (ohne oder mit minimaler Exploration),
    zeigt dabei grafische Ausgabe und misst den durchschnittlichen Reward.
    """
    # Exploration vorübergehend runtersetzen (oder policy_net auf eval stellen)
    old_eps_start = agent.eps_start
    agent.eps_start = 0.05  # Minimale Exploration im Test

    total_reward = 0.0
    for ep in range(episodes):
        env.reset()
        env.render()
        done = False
        ep_reward = 0.0

        # Hole initialen Zustand
        state_img = env.get_state()
        if not torch.is_tensor(state_img):
            state_img = torch.tensor(state_img, dtype=torch.float32)

        while not done:
            # Optional: env.render(), um es anzuzeigen
            env.render()

            action = agent.select_action(state_img)
            _, reward, done, _ = env.step(action)
            ep_reward += reward

            # Nächsten Bildzustand
            next_state_img = env.get_state()
            if not torch.is_tensor(next_state_img):
                next_state_img = torch.tensor(next_state_img, dtype=torch.float32)

            state_img = next_state_img

        print(f"Evaluierungs-Episode {ep+1}, Reward: {ep_reward:.2f}")
        total_reward += ep_reward

    avg_reward = total_reward / episodes
    print(f"Durchschnittlicher Reward nach {episodes} Episoden: {avg_reward:.2f}")

    # Exploration zurücksetzen
    agent.eps_start = old_eps_start


# =========================
# 6) Ausführen & Plotten
# =========================
if __name__ == "__main__":
    import pygame
    from Environment import BallOnBallEnv  # <--- Pfad anpassen

    env = BallOnBallEnv()

    # Trainiere Agent
    agent, rewards, losses = train_dqn(env, num_episodes=200)

    # Plotte Trainingsergebnisse
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

    # Nach dem Training: Evaluieren & grafisch anzeigen
    evaluate_agent(env, agent, episodes=5)

    env.close()
    pygame.quit()