"""Classes for implementing episilon annealing"""

class LinearSchedule(object):
    """Copy-pasted from:
       https://github.com/openai/baselines/blob/master/baselines/common/schedules.py
    """
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExpSchedule(object):
    def __init__(self, decay_rate, final_p, initial_p=1.0):
        """Exponential decay every time .step() is called
        Parameters
        ----------
        decay_rate: float
            exponential decay rate
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.epsilon = initial_p
        self.final_p = final_p
        self.decay_rate = decay_rate

    def value(self, t=None):
        """Takes `t`, a timestep, just to have identical
           signature to LinearSchedule.value
        """ 
        return self.epsilon

    def step(self):
        if self.epsilon > self.final_p:
            self.epsilon *= self.decay_rate