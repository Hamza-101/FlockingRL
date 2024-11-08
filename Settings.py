SimulationVariables = {
    "SimAgents" : 3,
    "AgentData" : [],
    "SafetyRadius" : 2, 
    "AccelerationInit" : 0.0,
    "NeighborhoodRadius" : 6, 
    "VelocityUpperLimit" : 2.5,
    "AccelerationUpperLimit" : 5.0,
    "X" : 2,
    "Y" : 2,
    "dt" : 0.1,
    "EvalTimeSteps" : 3000,
    "LearningTimeSteps" :  200000,
    "Episodes" : 20,
    "LearningRate": 0.003,
    "NumEnvs": 6,
    "Counter" : 0,
    "Seed" : 23,
    "ModelSeed" : 19,
}


Files = {
    "Flocking" : 'Results\Flocking',
    "VelocityBased" : 'Results\Velocitybased\Result_',
    "Reynolds" : 'Results\ReynoldRl\Result_',
    "Logs" : "Results\Flocking\Testing\Rewards\Components",
    "Positions" : "_positions",
    "Accelerations" : "_accelerations",
    "Velocities" : "_velocities",
    "Trajectory" : "_trajectory",
    "positions_directory" : "Test\Results\Flocking\Testing\Episodes"
}

Results = {
    "Sim" : "Simulation",
    "InitPositions" : "CorrectConfigs\Config_",
    "Positions" : "Results\Flocking\Testing\Episodes\Episode7_positions", 
    "Rewards" : "RewardTesting_Episode1",
    "EpRewards" : "Results",
    "Directory" : "CorrectConfigs",
    "EpisodalRewards" : "EpisodeRewards"
}


Animation = {
    "TimeSteps" : 3000
}