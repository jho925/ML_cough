from model import CoughClassifier, training, test, roc_auc
from train_test_val_split import test_dl, train_dl

def main():
	Model = CoughClassifier()
	training(Model, train_dl, 3)
	test(Model, test_dl)
	roc_auc(Model,test_dl)

if __name__ == '__main__':
	main()