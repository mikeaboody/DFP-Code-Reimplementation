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
