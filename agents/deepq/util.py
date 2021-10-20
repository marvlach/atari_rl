
class LinearSchedule(object):
	def __init__(self, initial_value, final_value, total_steps, warmup_steps):

		self.initial_value = initial_value
		self.final_value = final_value
		self.total_steps = total_steps
		self.warmup_steps = warmup_steps

		self.slope = -(self.initial_value - self.final_value) / self.total_steps
		self.intercept = self.initial_value - self.slope * self.warmup_steps

	def get_eps(self, step):
		if step < self.warmup_steps:
			return self.initial_value
		if step > self.total_steps + self.warmup_steps:
			return self.final_value
		return self.slope * step + self.intercept    

