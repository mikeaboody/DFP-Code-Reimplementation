import run_exp
import run_offline_exp
import doom_experience_serializing as ser
import sys

def main():
	assert len(sys.argv) > 2
	option = sys.argv[1]
	mode = sys.argv[2]

	if option == "serialize":
		assert len(sys.argv) > 5
		folder = sys.argv[3]
		chunk_size = int(sys.argv[4])
		num_steps = int(sys.argv[5])
		if mode == "mixed":
			ser.serialize_experiences_from_mixed_agent(folder, chunk_size, num_steps)
		elif mode == "random":
			ser.serialize_experiences_from_random_agent(folder, chunk_size, num_steps)
		else:
			ser.serialize_experiences_from_dfp_agent(folder, chunk_size, num_steps)
	elif option == "train":
		assert mode == "online"
		assert len(sys.argv) > 3
		run_exp.train(int(sys.argv[3]))
	elif option == "test":
		assert mode == "online"
		assert len(sys.argv) > 3
		run_exp.test(int(sys.argv[3]))
	elif option == "train_and_test":
		if mode == "online":
			run_exp.train_and_test()
		else:
			assert len(sys.argv) > 3
			run_offline_exp.train_and_test_offline(sys.argv[3])
	else:
		raise RuntimeError("Invalid option")

main()
