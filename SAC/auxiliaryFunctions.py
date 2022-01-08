import gym

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action