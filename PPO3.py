from stable_baselines import PPO2

class PPO3(PPO2):
    def setup_model(self):
        super().setup_model()
        self.env.env_method('train_network', 10)