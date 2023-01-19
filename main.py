from model import CoughClassifier, training, test
from train_test_val_split import test_dl, train_dl

def main():
	Model = CoughClassifier()
	training(Model, train_dl, 32)
	test(Model, test_dl)

if __name__ == '__main__':
	main()