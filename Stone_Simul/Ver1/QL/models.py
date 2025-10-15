# models.py
import pickle

def save_model(agent, filename):
    with open(filename, 'wb') as f:
        pickle.dump(agent.q_table, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        q_table = pickle.load(f)
    return q_table
