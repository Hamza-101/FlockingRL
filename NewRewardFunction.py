import os
import gym
import json
import numpy as np
import torch as th
from tqdm import tqdm
from Settings import *
from gym import spaces
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, get_system_info
from stable_baselines3.common.callbacks import BaseCallback
from scipy.signal import savgol_filter
import shutil
from PlotAnimationRL import *
import glob

th.cuda.is_available = lambda: True


positions_directory = "D:\\Thesis_\\FlockingFinal\\Results\\Flocking\\Testing\\Episodes"  

policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  
    net_arch=[dict(pi=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512], 
                   vf=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512])]  
)

class TQDMProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(TQDMProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.model.num_timesteps - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

class Agent:
    def __init__(self, position):
        
        self.position = np.array(position, dtype=float)

        self.acceleration = np.zeros(2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]
        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action):

        self.acceleration += action
        acc_magnitude = np.linalg.norm(self.acceleration)
        if acc_magnitude > 0:  
            if acc_magnitude > SimulationVariables["AccelerationUpperLimit"]:
                scaled_magnitude = SimulationVariables["AccelerationUpperLimit"] * np.tanh(acc_magnitude / SimulationVariables["AccelerationUpperLimit"])
                self.acceleration = (self.acceleration / acc_magnitude) * scaled_magnitude
        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > 0:
            if vel > self.max_velocity:
                self.velocity = self.velocity * np.tanh(self.max_velocity / vel)

        self.position += self.velocity * SimulationVariables["dt"]

        return self.position, self.velocity

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

class FlockingEnv(gym.Env):
    def __init__(self):
        # More episodes to train on
        super(FlockingEnv, self).__init__()
        self.episode=0
        self.counter=200
        self.CTDE=False
        self.current_timestep = 0
        self.reward_log = []
        self.np_random, _ = gym.utils.seeding.np_random(None)
        self.cumulative_rewards = {i: 0 for i in range(SimulationVariables["SimAgents"])}

        self.agents = [Agent(position) for position in self.read_agent_locations()]

        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        min_obs = np.array([-np.inf, -np.inf, -2.5, -2.5] * len(self.agents), dtype=np.float32)
        max_obs = np.array([np.inf, np.inf, 2.5, 2.5] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def step(self, actions):

        training_rewards = {}
        noisy_actions = actions + np.random.normal(loc=0, scale=0.5, size=actions.shape)
        actions = np.clip(noisy_actions, self.action_space.low, self.action_space.high)

        self.current_timestep += 1
        reward=0
        done=False
        info={}
        
        observations = self.simulate_agents(actions)
        reward, out_of_flock = self.calculate_reward() 

        if (self.CTDE==False):
            for agent in self.agents:
                if((self.check_collision(agent)) or (out_of_flock==True)):
                    done=True
                    env.reset()

        with open("training_rewards.json", "w") as f:
            json.dump(training_rewards, f)

        self.current_timestep = self.current_timestep + 1

        return observations, reward, done, info

    def reset(self):   
        env.seed(SimulationVariables["Seed"])
        self.agents = [Agent(position) for position in self.read_agent_locations()]

        print("\nInitial Agent Positions Reset:")
        for i, agent in enumerate(self.agents):
            print(f"Agent {i+1}: Position {agent.position}")

        for agent in self.agents:
            agent.acceleration = np.zeros(2)
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)             

        observation = self.get_observation().flatten()
        
        self.current_timestep = 0  
        return observation   

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):

        observations = []  

        actions_reshaped = actions.reshape(((SimulationVariables["SimAgents"]), 2))

        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions_reshaped[i])
            observation_pair = np.concatenate([position, velocity])
            observations = np.concatenate([observations, observation_pair])  

        return observations
    
    def check_collision(self, agent):
        for other in self.agents:
            if agent != other:
                distance = np.linalg.norm(agent.position - other.position)
                if distance < SimulationVariables["SafetyRadius"]:
                    return True
                
        return False

    def get_observation(self):
        observations = np.zeros((len(self.agents), 4), dtype=np.float32)
        
        for i, agent in enumerate(self.agents):
            observations[i] = [
                agent.position[0],
                agent.position[1],
                agent.velocity[0],
                agent.velocity[1]
            ]

        return observations
   
    def get_closest_neighbors(self, agent):

        neighbor_positions=[]
        neighbor_velocities=[]

        for _, other in enumerate(self.agents):
            if agent != other:
                distance = np.linalg.norm(other.position - agent.position)

                if(self.CTDE == True):
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)
                else:
                    neighbor_positions.append(other.position)
                    neighbor_velocities.append(other.velocity)

        return neighbor_positions, neighbor_velocities         
   
    def calculate_reward(self):
        total_reward = 0
        out_of_flock = False

        cumulative_alignment = 0
        cumulative_cohesion = 0

        for i, agent in enumerate(self.agents):
            neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)
            agent_reward, alignment_reward, cohesion_reward, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)

            self.cumulative_rewards[i] += agent_reward
            cumulative_alignment += alignment_reward
            cumulative_cohesion += cohesion_reward

            total_reward += agent_reward

        with open(os.path.join(positions_directory, f"CohesionRewardsEpisode{env.episode}.json"), "a") as f:
            f.write(f"{cumulative_cohesion} \n")
            
        with open(os.path.join(positions_directory, f"AlignmentRewardsEpisode{env.episode}.json"), "a") as f:
            f.write(f"{cumulative_alignment} \n")

        return total_reward, out_of_flock

    def reward(self, agent, neighbor_velocities, neighbor_positions):
            CohesionReward = 0
            AlignmentReward = 0
            total_reward = 0
            outofflock = False
                    # Define the midpoint
            midpoint = (SimulationVariables["SafetyRadius"] + SimulationVariables["NeighborhoodRadius"]) / 2

            if len(neighbor_positions) > 0:
                for neighbor_position in neighbor_positions:
                    distance = np.linalg.norm(agent.position - neighbor_position)

                    if SimulationVariables["SafetyRadius"] < distance < SimulationVariables["NeighborhoodRadius"]:
                        h = 1.125
                        k = 10  
                        a = -10 / ((SimulationVariables["NeighborhoodRadius"] - h) ** 2)  
                        CohesionReward += a * (distance - h) ** 2 + k

                    elif SimulationVariables["SafetyRadius"] < distance:
                        CohesionReward = 0

                    elif distance > SimulationVariables["NeighborhoodRadius"]:
                        CohesionReward = 0

                    average_velocity = np.mean(neighbor_velocities, axis = 0)
                    dot_product = np.dot(average_velocity, agent.velocity)
                    norm_product = np.linalg.norm(average_velocity) * np.linalg.norm(agent.velocity)

                    if norm_product == 0:
                        cos_angle = 1.0
                    else:
                        cos_angle = dot_product / norm_product

                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    orientation_diff = np.arccos(cos_angle)
                    alignment = (orientation_diff / np.pi)
                    AlignmentReward = -20 * alignment + 10  

            else:
                CohesionReward -= 10
                outofflock = True

            total_reward = CohesionReward + AlignmentReward      

            return total_reward, AlignmentReward, CohesionReward, outofflock

    def read_agent_locations(self):

        File = rf"{Results['InitPositions']}" + str(self.counter) + "\config.json"
        with open(File, "r") as f:
            data = json.load(f)

        return data

    def seed(self, seed=SimulationVariables["Seed"]):
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        return [seed]
#------------------------

def delete_files(): 
    Paths = [
        "Results/Flocking/Testing/Dynamics/Accelerations",
        "Results/Flocking/Testing/Dynamics/Velocities",
        "Results/Flocking/Testing/Rewards/Other"
    ]

    Logs = [
        "AlignmentReward_log.json", "CohesionReward_log.json",
        "SeparationReward_log.json", "CollisionReward_log.json",
        "Reward_Total_log.json"
    ]

    # Delete episode-specific JSON files in specified paths
    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    # Delete specific log files
    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    print("All specified files have been deleted.")

def generate_combined():
    """
    Generate and save plots for each episode as individual PNG files and a combined plot for all episodes.
    Plots include cohesion, alignment, combined rewards, and a secondary x-axis in seconds.
    """
    positions_directory = "Results/Flocking/Testing/Episodes"
    os.makedirs(positions_directory, exist_ok=True)  # Create directory for plots if it doesn't exist

    # Initialize Combined Plot
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))
    ax_combined_seconds = ax_combined.twiny()  # Add secondary x-axis for seconds

    # Process files for each episode
    cohesion_files = sorted([f for f in os.listdir(positions_directory) if f.startswith("CohesionRewardsEpisode")])
    alignment_files = sorted([f for f in os.listdir(positions_directory) if f.startswith("AlignmentRewardsEpisode")])

    for cohesion_file, alignment_file in zip(cohesion_files, alignment_files):
        episode = cohesion_file.split("CohesionRewardsEpisode")[1].split(".json")[0]

        # Load rewards from files and limit to 200 timesteps
        with open(os.path.join(positions_directory, cohesion_file), "r") as f:
            cohesion_rewards = [float(line.strip()) for line in f.readlines()][:200]
        with open(os.path.join(positions_directory, alignment_file), "r") as f:
            alignment_rewards = [float(line.strip()) for line in f.readlines()][:200]

        # Calculate combined rewards
        combined_rewards = [c + a for c, a in zip(cohesion_rewards, alignment_rewards)]

        timesteps = range(1, len(cohesion_rewards) + 1)
        seconds = [timestep / 10 for timestep in timesteps]  # Convert timesteps to seconds (10 timesteps = 1 second)

        # Individual Episode Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax_seconds = ax.twiny()  # Add secondary x-axis for seconds

        # Plot cohesion, alignment, and combined rewards
        ax.plot(timesteps, cohesion_rewards, label=f"Cohesion (Episode {episode})", alpha=0.7, color="blue")
        ax.plot(timesteps, alignment_rewards, label=f"Alignment (Episode {episode})", alpha=0.7, color="green")
        ax.plot(timesteps, combined_rewards, label=f"Combined (Episode {episode})", alpha=0.7, color="orange")

        # Format the primary x-axis (timesteps)
        ax.set_title(f"Rewards for Episode {episode}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True)

        # Format the secondary x-axis (seconds)
        ax.set_xlim(1, 200)
        ax_seconds.set_xlim(ax.get_xlim())
        ax_seconds.set_xticks(ax.get_xticks())
        ax_seconds.set_xticklabels([f"{tick / 10:.1f}" for tick in ax.get_xticks()])
        ax_seconds.set_xlabel("Time (seconds)")

        # Save Individual Episode Plot
        plt.tight_layout()
        plt.savefig(os.path.join(positions_directory, f"Episode_{episode}_Rewards.png"), dpi=300)
        plt.close(fig)

        # Add to Combined Plot
        ax_combined.plot(timesteps, cohesion_rewards, label=f"Cohesion (Episode {episode})", alpha=0.5)
        ax_combined.plot(timesteps, alignment_rewards, label=f"Alignment (Episode {episode})", alpha=0.5)
        ax_combined.plot(timesteps, combined_rewards, label=f"Combined (Episode {episode})", alpha=0.5)

    # Format Combined Plot
    ax_combined.set_title("Cohesion, Alignment, and Combined Rewards - All Episodes (200 Timesteps)")
    ax_combined.set_xlabel("Timestep")
    ax_combined.set_ylabel("Reward")
    ax_combined.legend()
    ax_combined.grid(True)

    # Format the secondary x-axis (seconds) for the combined plot
    ax_combined.set_xlim(1, 200)
    ax_combined_seconds.set_xlim(ax_combined.get_xlim())
    ax_combined_seconds.set_xticks(ax_combined.get_xticks())
    ax_combined_seconds.set_xticklabels([f"{tick / 10:.1f}" for tick in ax_combined.get_xticks()])
    ax_combined_seconds.set_xlabel("Time (seconds)")

    # Save Combined Plot
    plt.tight_layout()
    combined_plot_path = os.path.join(positions_directory, "Combined_Cohesion_Alignment_Rewards.png")
    plt.savefig(combined_plot_path, dpi=300)
    plt.close(fig_combined)

    print(f"Plots saved in directory: {positions_directory}")
    print(f"Combined plot saved at: {combined_plot_path}")

def setup_episode_folder(episode_name):
    episode_folder = os.path.join(positions_directory, episode_name)
    if os.path.exists(episode_folder):
        for file in os.listdir(episode_folder):
            os.remove(os.path.join(episode_folder, file))  # Remove existing files in the folder
    else:
        os.makedirs(episode_folder, exist_ok=True)
    return episode_folder

positions_directory = "D:\\Thesis_\\FlockingFinal\\Results\\Flocking\\Testing\\Episodes" 

def generateVelocity(episode, episode_folder):
    velocities_dict = {}

    velocity_file_path = os.path.join(positions_directory, f"Episode{episode}_velocities.json")
    
    if not os.path.exists(velocity_file_path):
        print(f"File {velocity_file_path} not found.")
        return

    with open(velocity_file_path, 'r') as f:
        episode_velocities = json.load(f)

    for agent_id in range(3):
        velocities_dict.setdefault(agent_id, []).extend(episode_velocities.get(str(agent_id), []))

    colors = ['blue', 'orange', 'green']  

    downsample_factor = 10  
    for agent_id in range(3):
        plt.figure(figsize=(10, 5))
        plt.clf()

        agent_velocities = np.array(velocities_dict[agent_id])
        agent_velocities = savgol_filter(agent_velocities, window_length=3, polyorder=2, axis=0)  
        velocities_magnitude = np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)
        
        plt.plot(velocities_magnitude[::downsample_factor], label=f"Agent {agent_id+1}", color=colors[agent_id], linewidth=0.5)
        
        plt.title(f"Velocity - Episode {episode} - Agent {agent_id+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity Magnitude")
        plt.ylim([0, 5])  
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(episode_folder, f"Agent_{agent_id+1}_Velocity.png"))
        plt.close()  
        print(f"Velocity plot saved for Episode {episode}, Agent {agent_id+1}")

def generateAcceleration(episode, episode_folder):
    acceleration_file_path = os.path.join(positions_directory, f"Episode{episode}_accelerations.json")
    
    if not os.path.exists(acceleration_file_path):
        print(f"File {acceleration_file_path} not found.")
        return

    with open(acceleration_file_path, 'r') as f:
        episode_accelerations = json.load(f)

    colors = ['blue', 'orange', 'green']  

    downsample_factor = 10  
    for agent_id in range(3):
        plt.figure(figsize=(10, 5))
        plt.clf()

        agent_accelerations = np.array(episode_accelerations[str(agent_id)])
        smoothed_accelerations = np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)
        smoothed_accelerations = savgol_filter(smoothed_accelerations, window_length=15, polyorder=3, axis=0)  

        plt.plot(smoothed_accelerations[::downsample_factor], label=f"Agent {agent_id+1}", color=colors[agent_id], linewidth=0.5)

        plt.title(f"Acceleration - Episode {episode} - Agent {agent_id+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Acceleration Magnitude")
        plt.ylim([0, 10])  
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(episode_folder, f"Agent_{agent_id+1}_Acceleration.png"))
        plt.close()  
        print(f"Acceleration plot saved for Episode {episode}, Agent {agent_id+1}")

def generatePlots():
    for episode in range(SimulationVariables["Episodes"]):
        episode_name = f"Episode{episode}".split('_')[0]
        
        episode_folder = setup_episode_folder(episode_name)
        
        generateVelocity(episode, episode_folder)
        generateAcceleration(episode, episode_folder)

# Ensure all relevant files are deleted at the start of execution
def delete_existing_files(directory, pattern):
    files = glob.glob(os.path.join(directory, pattern))
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
#------------------------
class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.loss_threshold = 2000

    def _on_step(self) -> bool:
        
        if len(self.model.ep_info_buffer) >= 1000:
            recent_losses = [ep_info['loss'] for ep_info in self.model.ep_info_buffer[-1000:]]
            average_loss = np.mean(recent_losses)

            if average_loss < self.loss_threshold:
                print(f"Stopping training because average loss ({average_loss}) is below threshold.")
                return False  

        return True

class AdaptiveExplorationCallback(BaseCallback):
    def __init__(self, initial_ent_coef=0.15, min_ent_coef=1e-10, decay_rate=0.95, max_reward_threshold=20, verbose=0):
        super(AdaptiveExplorationCallback, self).__init__(verbose)
        self.initial_ent_coef = initial_ent_coef       
        self.min_ent_coef = min_ent_coef               
        self.decay_rate = decay_rate                   
        self.ent_coef = initial_ent_coef               
        self.max_reward_threshold = max_reward_threshold  

    def _on_training_start(self):
        self.model.ent_coef = self.initial_ent_coef

    def _on_step(self) -> bool:
        cumulative_rewards = self.model.env.get_attr('cumulative_rewards')[0] 
        all_agents_above_threshold = all(reward >= self.max_reward_threshold for reward in cumulative_rewards.values())
        if all_agents_above_threshold:
            self.ent_coef = max(self.ent_coef * self.decay_rate, self.min_ent_coef)
        else:
            self.ent_coef = self.initial_ent_coef
        self.model.ent_coef = self.ent_coef

        return True
#------------------------

if os.path.exists(Results["Rewards"]):
    os.remove(Results["Rewards"])
    print(f"File {Results['Rewards']} has been deleted.")

if os.path.exists("training_rewards.json"):
    os.remove("training_rewards.json")
    print(f"File training_rewards has been deleted.")    

def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    env.seed(seed)
    env.action_space.seed(seed)

env = FlockingEnv()
seed_everything(SimulationVariables["Seed"])

loss_callback = LossCallback()
adaptive_exploration_callback = AdaptiveExplorationCallback()
progress_callback = TQDMProgressCallback(total_timesteps=SimulationVariables["LearningTimeSteps"])

get_system_info()
print(env.action_space.shape)
print(env.observation_space.shape)


#Model Training
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)
model.set_random_seed(SimulationVariables["ModelSeed"])

model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"],  callback=[progress_callback, adaptive_exploration_callback])
model.save(rf"{Files['Flocking']}\\Models\\FlockingCombinedNew")

velocities_for_episode_2 = np.array([[0.0, 2.5], [-2.5, -2.5], [2.5, 0.0]])
velocities_for_episode_3 = np.array([[2.5, 0.0], [-2.5, 0.0], [-2.5, 1.0]])

env = FlockingEnv()
model = PPO.load(rf'{Files["Flocking"]}\Models\FlockingCombinedNew')

delete_files()
positions_directory = rf"{Files['Flocking']}/Testing/Episodes/"
os.makedirs(positions_directory, exist_ok=True)

env.counter=0
episode_rewards_dict = {}
positions_dict = {i: [] for i in range(len(env.agents))}

delete_existing_files(positions_directory, "CohesionRewardsEpisode*.json")
delete_existing_files(positions_directory, "AlignmentRewardsEpisode*.json")

for episode in tqdm(range(0, SimulationVariables["Episodes"])):
    env.episode = episode
    obs = env.reset()
    env.CTDE = True
    done = False
    timestep = 0
    reward_episode = []

    distances_dict = []
    positions_dict = {i: [] for i in range(len(env.agents))}
    velocities_dict = {i: [] for i in range(len(env.agents))}
    accelerations_dict = {i: [] for i in range(len(env.agents))}
    trajectory_dict = {i: [] for i in range(len(env.agents))}
    
    print(f"\n--- Episode {episode} ---")  
    print(env.counter)

    if episode == 1:  
        for i, agent in enumerate(env.agents):
            agent.velocity = velocities_for_episode_2[i]
            print(f"Agent {i+1} initial velocity for Episode 2: {agent.velocity}")
    elif episode == 2:  # Third episode
        for i, agent in enumerate(env.agents):
            agent.velocity = velocities_for_episode_3[i]
            print(f"Agent {i+1} initial velocity for Episode 3: {agent.velocity}")


    for i, agent in enumerate(env.agents):
        accelerations_dict[i].append(agent.acceleration.tolist())
        velocities_dict[i].append(agent.velocity.tolist())
        positions_dict[i].append(agent.position.tolist())
        trajectory_dict[i].append(agent.position.tolist())

    while timestep < SimulationVariables["EvalTimeSteps"]:
        actions, state = model.predict(obs)
        obs, reward, done, info = env.step(actions)
        reward_episode.append(reward)
        
        timestep_distances = {}  
        
        for i, agent in enumerate(env.agents):
            positions_dict[i].append(agent.position.tolist())
            velocity = agent.velocity.tolist()
            velocities_dict[i].append(velocity)
            acceleration = agent.acceleration.tolist()
            accelerations_dict[i].append(acceleration)
            trajectory_dict[i].append(agent.position.tolist())
            
            distances = []
            for j, other_agent in enumerate(env.agents):
                if i != j:  
                    distance = np.linalg.norm(np.array(other_agent.position) - np.array(agent.position))
                    distances.append(distance)
            timestep_distances[i] = distances

        distances_dict.append(timestep_distances)

        timestep += 1
        episode_rewards_dict[str(episode)] = reward_episode

    with open(os.path.join(positions_directory, f"Episode{episode}_positions.json"), 'w') as f:
        json.dump(positions_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'w') as f:
        json.dump(velocities_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'w') as f:
        json.dump(accelerations_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_distances.json"), 'w') as f:
        json.dump(distances_dict, f, indent=4)  
    with open(os.path.join(positions_directory, f"Episode{episode}_trajectory.json"), 'w') as f:
        json.dump(trajectory_dict, f, indent=4)

    env.counter += 1
    print(sum(reward_episode))

with open(rf"{Results['EpisodalRewards']}.json", 'w') as f:
    json.dump(episode_rewards_dict, f, indent=4)

env.close()
print("Testing completed")

generatePlots()
generate_combined()
