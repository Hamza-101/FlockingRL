
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
    

# Ensure the model only uses CPU
# Ensure the model only uses CPU
th.cuda.is_available = lambda: True

positions_directory = "D:\\Thesis_\\FlockingFinal\\Results\\Flocking\\Testing\\Episodes"  # Update this to the correct directory

# Neural Network Parameters
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  # Using ReLU activation function
    net_arch=[dict(pi=[512, 512, 512, 512, 512, 512, 512, 512], 
                   vf=[512, 512, 512, 512, 512, 512, 512, 512])]  # Separate networks for policy (pi) and value function (vf)
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
            # Update progress bar with the number of timesteps
            self.pbar.update(self.model.num_timesteps - self.pbar.n)
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()

#Multiple random initialization
class Agent:
    def __init__(self, position):
        
        self.position = np.array(position, dtype=float)

        # Random initialization of velocity and initializing acceleration to null
        self.acceleration = np.zeros(2)
        self.max_acceleration = SimulationVariables["AccelerationUpperLimit"]

        self.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)       
        self.max_velocity = SimulationVariables["VelocityUpperLimit"]

    def update(self, action, CTDE):

        ###
        self.acceleration += action
        self.acceleration = SimulationVariables["AccelerationUpperLimit"] * np.tanh(self.acceleration / SimulationVariables["AccelerationUpperLimit"])
        self.velocity += self.acceleration * SimulationVariables["dt"]
        vel = np.linalg.norm(self.velocity)
        if vel > self.max_velocity:
            self.velocity = (self.velocity / vel) * self.max_velocity   
        self.position += self.velocity * SimulationVariables["dt"]

        return self.position, self.velocity

class Encoder(json.JSONEncoder):
    def default(self, obj):
        return json.JSONEncoder.default(self, obj)  

# 3 Agents
class FlockingEnv(gym.Env):
    def __init__(self):

        super(FlockingEnv, self).__init__()
        self.episode=0
        self.counter=3602
        self.CTDE=False
        self.current_timestep = 0
        self.reward_log = []
        self.np_random, _ = gym.utils.seeding.np_random(None)
        self.cumulative_rewards = {i: 0 for i in range(SimulationVariables["SimAgents"])}

        self.agents = [Agent(position) for position in self.read_agent_locations()]

        # Use settings file in actions and observations
        min_action = np.array([-5, -5] * len(self.agents), dtype=np.float32)
        max_action = np.array([5, 5] * len(self.agents), dtype=np.float32)
        self.action_space = spaces.Box(low=min_action, high=max_action, dtype=np.float32)

        #Check this
        min_obs = np.array([-np.inf, -np.inf, -2.5, -2.5] * len(self.agents), dtype=np.float32)
        max_obs = np.array([np.inf, np.inf, 2.5, 2.5] * len(self.agents), dtype=np.float32)
        self.observation_space = spaces.Box(low=min_obs, high=max_obs, dtype=np.float32)

    def step(self, actions):

        training_rewards = {}
        #REM
        noisy_actions = actions + np.random.normal(loc=0, scale=0.5, size=actions.shape)
        actions = np.clip(noisy_actions, self.action_space.low, self.action_space.high)

        self.current_timestep += 1
        reward=0
        done=False
        info={}
        
        #Noisy Actions
        observations = self.simulate_agents(actions)
        reward, out_of_flock = self.calculate_reward() # Adjust this if each agent's reward is calculated separately

        if (self.CTDE==False):
            for agent in self.agents:
                if((self.check_collision(agent)) or (out_of_flock==True)):
                    done=True
                    env.reset()

        #Check position
        with open("training_rewards.json", "w") as f:
            json.dump(training_rewards, f)

        self.current_timestep = self.current_timestep + 1

        return observations, reward, done, info

    def reset(self):
        
        env.seed(SimulationVariables["Seed"])

        self.agents = [Agent(position) for position in self.read_agent_locations()]

        for agent in self.agents:
            agent.acceleration = np.zeros(2)
            agent.velocity = np.round(np.random.uniform(-SimulationVariables["VelocityUpperLimit"], SimulationVariables["VelocityUpperLimit"], size=2), 2)             

        observation = self.get_observation().flatten()
        
        ################################
        self.current_timestep = 0  # Reset time step count
        ################################
        return observation   

    def close(self):
        print("Environment is closed. Cleanup complete.")
        
    def simulate_agents(self, actions):

        observations = []  # Initialize an empty 1D array

        actions_reshaped = actions.reshape(((SimulationVariables["SimAgents"]), 2))

        for i, agent in enumerate(self.agents):
            position, velocity = agent.update(actions_reshaped[i], self.CTDE)
            observation_pair = np.concatenate([position, velocity])
            observations = np.concatenate([observations, observation_pair])  # Concatenate each pair directly

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

        # Reshape the observation into 1D                    
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
        Collisions = {}
        out_of_flock = False

        # Initialize cumulative rewards for each agent
        for idx, _ in enumerate(self.agents):
            Collisions[idx] = []
        
        for i, agent in enumerate(self.agents): 
            neighbor_positions, neighbor_velocities = self.get_closest_neighbors(agent)
            agent_reward, out_of_flock = self.reward(agent, neighbor_velocities, neighbor_positions)
            # Add this agent's reward to its cumulative total
            self.cumulative_rewards[i] += agent_reward
            total_reward += agent_reward

        return total_reward, out_of_flock

    def reward(self, agent, neighbor_velocities, neighbor_positions):
            CohesionReward = 0
            AlignmentReward = 0
            total_reward = 0
            outofflock = False
            midpoint = 4

            if len(neighbor_positions) > 0:
                for neighbor_position in neighbor_positions:
                    distance = np.linalg.norm(agent.position - neighbor_position)

                    if distance <= SimulationVariables["SafetyRadius"]:
                        CohesionReward += 0
                        
                    elif SimulationVariables["SafetyRadius"] < distance <= midpoint:
                        CohesionReward += (10 / (midpoint - SimulationVariables["SafetyRadius"])) * (distance - SimulationVariables["SafetyRadius"])

                    elif midpoint < distance <= SimulationVariables["NeighborhoodRadius"]:
                        CohesionReward += 10 - (10 / (SimulationVariables["NeighborhoodRadius"] - midpoint)) * (distance - midpoint)
        
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

            return total_reward, outofflock

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
    Paths = ["Results\Flocking\Testing\Dynamics\Accelerations", "Results\Flocking\Testing\Dynamics\Velocities", 
            "Results\Flocking\Testing\Rewards\Other"]

    Logs = ["AlignmentReward_log.json", "CohesionReward_log.json",
            "SeparationReward_log.json", "CollisionReward_log.json",
            "Reward_Total_log.json"]

    for Path in Paths:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], Path, f"Episode{episode}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")

    for log_file in Logs:
        for episode in range(0, 10):
            file_path = os.path.join(Files['Flocking'], "Testing", "Rewards", "Components", f"Episode{episode}", log_file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File {file_path} has been deleted.")       

def generateCombined():
    with open(rf"{Results['EpisodalRewards']}.json", "r") as f:
        episode_rewards_dict = json.load(f)

    keys_above_threshold = []
    keys_below_threshold = []

    for episode, rewards in episode_rewards_dict.items():
        total_sum = sum(rewards)
        if total_sum > 1000000:
            keys_above_threshold.append(episode)
        else:
            keys_below_threshold.append(episode)

    plt.figure(figsize=(10, 6))
    plt.clf()

    #Fix this
    for episode in keys_above_threshold:
        rewards = episode_rewards_dict[episode]
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Episode {episode}", alpha=0.7)

    for episode in keys_below_threshold:
        rewards = episode_rewards_dict[episode]
        plt.plot(range(1, len(rewards) + 1), rewards, label=f"Episode {episode}", alpha=0.7)

    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    plt.title(f"Rewards for Episodes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Output.png", dpi=300)


def setup_episode_folder(episode_name):
    # Create (or recreate) folder for this episode
    episode_folder = os.path.join(positions_directory, episode_name)
    if os.path.exists(episode_folder):
        for file in os.listdir(episode_folder):
            os.remove(os.path.join(episode_folder, file))  # Remove existing files in the folder
    else:
        os.makedirs(episode_folder, exist_ok=True)
    return episode_folder

positions_directory = "D:\\Thesis_\\FlockingFinal\\Results\\Flocking\\Testing\\Episodes"  # Update this to the correct directory

def generateVelocity(episode, episode_folder):
    velocities_dict = {}

    # File path for velocity data
    velocity_file_path = os.path.join(positions_directory, f"Episode{episode}_velocities.json")
    
    # Check if the file exists to avoid FileNotFoundError
    if not os.path.exists(velocity_file_path):
        print(f"File {velocity_file_path} not found.")
        return

    # Load velocity data from JSON
    with open(velocity_file_path, 'r') as f:
        episode_velocities = json.load(f)

    # Organize velocities by agent
    for agent_id in range(3):
        velocities_dict.setdefault(agent_id, []).extend(episode_velocities.get(str(agent_id), []))

    # Define colors for each agent
    colors = ['blue', 'orange', 'green']  # Add more colors if needed

    # Generate and save a separate plot for each agent with specified colors
    downsample_factor = 10  # Adjust as needed
    for agent_id in range(3):
        plt.figure(figsize=(10, 5))
        plt.clf()

        agent_velocities = np.array(velocities_dict[agent_id])
        agent_velocities = savgol_filter(agent_velocities, window_length=3, polyorder=2, axis=0)  # Increased smoothing
        velocities_magnitude = np.sqrt(agent_velocities[:, 0]**2 + agent_velocities[:, 1]**2)
        
        # Plot downsampled data for the agent with specified color and thinner line
        plt.plot(velocities_magnitude[::downsample_factor], label=f"Agent {agent_id+1}", color=colors[agent_id], marker='o', markersize=3, linewidth=0.5)
        
        plt.title(f"Velocity - Episode {episode} - Agent {agent_id+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Velocity Magnitude")
        plt.ylim([0, 5])  # Limit y-axis range to reduce outliers
        plt.legend()
        plt.grid(True)
        
        # Save plot with agent-specific filename
        plt.savefig(os.path.join(episode_folder, f"Agent_{agent_id+1}_Velocity.png"))
        plt.close()  # Close the figure to free memory
        print(f"Velocity plot saved for Episode {episode}, Agent {agent_id+1}")

def generateAcceleration(episode, episode_folder):
    # File path for acceleration data
    acceleration_file_path = os.path.join(positions_directory, f"Episode{episode}_accelerations.json")
    
    # Check if the file exists to avoid FileNotFoundError
    if not os.path.exists(acceleration_file_path):
        print(f"File {acceleration_file_path} not found.")
        return

    # Load acceleration data from JSON
    with open(acceleration_file_path, 'r') as f:
        episode_accelerations = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.clf()

    # Plot accelerations for each agent with increased smoothing and downsampling
    downsample_factor = 10  # Adjust as needed
    for agent_id in range(3):
        agent_accelerations = np.array(episode_accelerations[str(agent_id)])
        smoothed_accelerations = np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)
        smoothed_accelerations = savgol_filter(smoothed_accelerations, window_length=15, polyorder=3, axis=0)  # Increased smoothing

        # Plot downsampled data
        plt.plot(smoothed_accelerations[::downsample_factor], label=f"Agent {agent_id+1}")

    plt.title(f"Acceleration - Episode {episode}")
    plt.xlabel("Time Step")
    plt.ylabel("Acceleration Magnitude")
    plt.ylim([0, 10])  # Limit y-axis range to reduce outliers
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(episode_folder, "Acceleration.png"))
    print(f"Acceleration plot saved for Episode {episode}")


def generateAcceleration(episode, episode_folder):
    # File path for acceleration data
    acceleration_file_path = os.path.join(positions_directory, f"Episode{episode}_accelerations.json")
    
    # Check if the file exists to avoid FileNotFoundError
    if not os.path.exists(acceleration_file_path):
        print(f"File {acceleration_file_path} not found.")
        return

    # Load acceleration data from JSON
    with open(acceleration_file_path, 'r') as f:
        episode_accelerations = json.load(f)

    # Define colors for each agent (optional for visual distinction)
    colors = ['blue', 'orange', 'green']  # Add more colors if needed

    # Generate and save a separate plot for each agent
    downsample_factor = 10  # Adjust as needed
    for agent_id in range(3):
        plt.figure(figsize=(10, 5))
        plt.clf()

        agent_accelerations = np.array(episode_accelerations[str(agent_id)])
        smoothed_accelerations = np.sqrt(agent_accelerations[:, 0]**2 + agent_accelerations[:, 1]**2)
        smoothed_accelerations = savgol_filter(smoothed_accelerations, window_length=15, polyorder=3, axis=0)  # Increased smoothing

        # Plot downsampled data with a thinner line
        plt.plot(smoothed_accelerations[::downsample_factor], label=f"Agent {agent_id+1}", color=colors[agent_id], linewidth=0.5)

        plt.title(f"Acceleration - Episode {episode} - Agent {agent_id+1}")
        plt.xlabel("Time Step")
        plt.ylabel("Acceleration Magnitude")
        plt.ylim([0, 10])  # Limit y-axis range to reduce outliers
        plt.legend()
        plt.grid(True)

        # Save plot with agent-specific filename
        plt.savefig(os.path.join(episode_folder, f"Agent_{agent_id+1}_Acceleration.png"))
        plt.close()  # Close the figure to free memory
        print(f"Acceleration plot saved for Episode {episode}, Agent {agent_id+1}")

def generatePlots():
    for episode in range(SimulationVariables["Episodes"]):
        episode_name = f"Episode{episode}".split('_')[0]
        
        # Setup the folder for this episode
        episode_folder = setup_episode_folder(episode_name)
        
        # Generate velocity and acceleration plots
        generateVelocity(episode, episode_folder)
        generateAcceleration(episode, episode_folder)
#------------------------
class LossCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(LossCallback, self).__init__(verbose)
        self.loss_threshold = 2000

    def _on_step(self) -> bool:
        
        # if(self.current_timestep > (SimulationVariables["LearningTimesteps"]/2)):
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

env=FlockingEnv()
seed_everything(SimulationVariables["Seed"])

loss_callback = LossCallback()
adaptive_exploration_callback = AdaptiveExplorationCallback()
progress_callback = TQDMProgressCallback(total_timesteps=SimulationVariables["LearningTimeSteps"])

get_system_info()
print(env.action_space.shape)
print(env.observation_space.shape)

# Model Training
# model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_Agents_tensorboard/", verbose=1)
# model.set_random_seed(SimulationVariables["ModelSeed"])

# model.learn(total_timesteps=SimulationVariables["LearningTimeSteps"],  callback=[progress_callback, adaptive_exploration_callback])
# model.save(rf"{Files['Flocking']}\\Models\\FlockingCombinedNew")

env = FlockingEnv()
model = PPO.load(rf'{Files["Flocking"]}\Models\FlockingCombinedNew')

delete_files()
positions_directory = rf"{Files['Flocking']}/Testing/Episodes/"
os.makedirs(positions_directory, exist_ok=True)

env.counter=389
episode_rewards_dict = {}
positions_dict = {i: [] for i in range(len(env.agents))}

for episode in tqdm(range(0, SimulationVariables['Episodes'])):
    env.episode = episode
    print("Episode:", episode)
    env.CTDE = True
    obs = env.reset()
    done = False
    timestep = 0
    reward_episode = []

    # Initialize dictionaries to store data
    positions_dict = {i: [] for i in range(len(env.agents))}
    velocities_dict = {i: [] for i in range(len(env.agents))}
    accelerations_dict = {i: [] for i in range(len(env.agents))}
    trajectory_dict = {i: [] for i in range(len(env.agents))}

    while timestep < min(SimulationVariables["EvalTimeSteps"], 5000):
        actions, state = model.predict(obs)
        obs, reward, done, info = env.step(actions)
        reward_episode.append(reward)
        
        for i, agent in enumerate(env.agents):

            positions_dict[i].append(agent.position.tolist())
            velocity = agent.velocity.tolist()
            velocities_dict[i].append(velocity)
            acceleration = agent.acceleration.tolist()
            accelerations_dict[i].append(acceleration)
            trajectory_dict[i].append(agent.position.tolist())

        timestep += 1
        episode_rewards_dict[str(episode)] = reward_episode

    with open(os.path.join(positions_directory, f"Episode{episode}_positions.json"), 'w') as f:
        json.dump(positions_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_velocities.json"), 'w') as f:
        json.dump(velocities_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_accelerations.json"), 'w') as f:
        json.dump(accelerations_dict, f, indent=4)
    with open(os.path.join(positions_directory, f"Episode{episode}_trajectory.json"), 'w') as f:
        json.dump(trajectory_dict, f, indent=4)

    env.counter += 1
    print(sum(reward_episode))
    

with open(rf"{Results['EpisodalRewards']}.json", 'w') as f:
    json.dump(episode_rewards_dict, f, indent=4)

env.close()
print("Testing completed")


generatePlots()
generateCombined()
