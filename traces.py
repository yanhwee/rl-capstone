import numpy as np

class Trace:
    def __call__(self, agent, state, action):
        raise NotImplementedError

class AccumulatingTrace(Trace):
    def __call__(self, agent, state, action):
        traces = agent.traces
        traces *= agent.discount_factor * agent.trace_decay
        traces[action] += state

class ReplacingTrace(Trace):
    def __call__(self, agent, state, action):
        traces = agent.traces
        traces *= agent.discount_factor * agent.trace_decay
        traces[action, state == 1] = 1

class DutchTrace(Trace):
    def update_trace(self, agent, state, action):
        traces = agent.traces
        traces *= agent.discount_factor * agent.trace_decay
        traces += state * (1 - np.dot(agent.learning_rate * traces, state))
    def update_value(self, agent, state, action, td_error):
        traces = agent.traces
        weights = agent.weights
        prev_weights = weights.copy()
        agent.prev_weights = weights.copy()
        weights += agent.learning_rate * (
            td_error * traces + (
                np.dot(weights, state) -
                np.dot()
            )
        )