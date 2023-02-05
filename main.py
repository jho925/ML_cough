from model import CoughClassifier, training, test, get_dataloader
import torch

def main():
	Model = CoughClassifier()
	train_dl, test_dl = get_dataloader('ML_labels.csv',0)
	training(Model, train_dl,test_dl, 30)
	torch.save(Model.state_dict(), 'model.pth')
	test(Model, test_dl)
	# roc_auc(Model,test_dl)

if __name__ == '__main__':
	main()