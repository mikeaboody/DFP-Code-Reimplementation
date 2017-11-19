from abstraction import Experience
from util import *
import shutil
import os

class ExperienceSerializer(object):
	def __init__(self, folder, chunk_size, wipe=True):
		self.num_exp_serialized = 0
		self.folder = folder
		self.chunk_size = chunk_size
		self.exp_lst = []
		if wipe:
			try:
				shutil.rmtree(folder)
			except OSError:
				pass
			os.makedirs(folder)

	def serialize_experience(self, exp):
		self.exp_lst.append(exp)
		self.num_exp_serialized += 1
		if self.num_exp_serialized % self.chunk_size == 0:
			self.flush()
	def flush(self):
		if len(self.exp_lst) != 0:
			filename = str(self.num_exp_serialized) + ".npy"
			path = os.path.join(self.folder, filename)
			serialize_experiences(self.exp_lst, path)
			self.exp_lst = []


class ExperienceDeserializer(object):
	def __init__(self, folder):
		self.folder = folder
		self.file_names = os.listdir(folder)
	def deserialized_generator(self):
		curr_file_index = 0
		curr_experience_lst = None
		while curr_file_index < len(self.file_names) or curr_experience_lst is not None:
			if curr_experience_lst is None:
				filename = self.file_names[curr_file_index]
				path = os.path.join(self.folder, filename)
				curr_file_index += 1
				curr_experience_lst = deserialize_experiences(path)
				exp_lst_index = 0
			if exp_lst_index < len(curr_experience_lst):
				yield curr_experience_lst[exp_lst_index]
				exp_lst_index += 1
			else:
				curr_experience_lst = None
def create_experience():
	sens = np.random.random_sample(size=(84,84,1))
	meas = np.random.random_sample(size=(1,))
	goal = np.random.random_sample(size=(6,))
	# last value indicate index of action
	label = np.random.random_sample(size=(6,))
	obs = Observation(sens, meas)
	exp = Experience(obs, action_to_one_hot([0,1,0]), goal, label)
	return exp

# size = 49999
# lst = [create_experience() for _ in range(size)]
# exp_ser = ExperienceSerializer("exp", 10000)
# for exp in lst:
# 	exp_ser.serialize_experience(exp)
# exp_ser.flush()
# exp_deser = ExperienceDeserializer("exp")
# gen = exp_deser.deserialized_generator()

# # serialize_experiences(lst, "exp.npy")
# # deserialized_lst = deserialize_experiences("exp.npy")
# passs = True
# for i in range(len(lst)):
# 	exp = lst[i]
# 	copy = gen.next()
# 	# same = np.array_equal(exp.sens(), copy.sens()) and np.array_equal(exp.meas(), copy.meas()) and np.array_equal(exp.action(), copy.action()) and np.array_equal(exp.goal(), copy.goal()) and np.array_equal(exp.label(), copy.label())
# 	same = exp == copy
# 	if not same:
# 		print("BAD")
# 		passs = True
# print(passs)

# import pdb; pdb.set_trace()
