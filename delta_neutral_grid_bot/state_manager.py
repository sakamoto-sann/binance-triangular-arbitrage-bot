import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StateManager:
    def __init__(self, state_file='bot_state.json'):
        self.state_file = state_file
        self.state = self.load_state()

    def load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.info("No state file found, starting with a fresh state.")
            return {'grid_orders': []}
        except json.JSONDecodeError:
            logging.error("Could not decode state file, starting with a fresh state.")
            return {'grid_orders': []}

    def save_state(self, state):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logging.error(f"Could not save state: {e}")

    def get(self, key, default=None):
        return self.state.get(key, default)

    def set(self, key, value):
        self.state[key] = value
        self.save_state(self.state)