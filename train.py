# python3 train.py --dataset ../../Datasets/kvasir_224_resized/kvasir --batch_size 2 --epochs 1

import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout

from model.resnet import ResNet18
from model.wide_resnet import WideResNet

from first import *
import torchvision.models as trained_models
import pandas as pd
import extras as extras

import config as args

def  main():
    def test(loader):
        cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
        correct = 0.
        total = 0.
        for images, labels in loader:
            images = images.cuda()
            labels = labels.cuda()

            with torch.no_grad():
                pred = cnn(images)

            pred = torch.max(pred.data, 1)[1]
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        val_acc = correct / total
        cnn.train()
        return val_acc

    result_df = pd.DataFrame(columns = ['Test_Acc', 'Test, Pre', 'Test_Re', 'Test_F1', 'Train_Acc',
    'Train_Pre', 'Train_Re', 'Train_F1'])

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    test_id = args.dataset.split("/")[-1] + '_' + args.model

    train_transform, test_transform = get_base_transform(224)
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    dataset = args.dataset
    train_dataset, test_dataset = extras.get_train_test_dataset(dataset, train_transform, test_transform)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)



    create_dir('logs/', False)
    create_dir('checkpoints/', False)


    # Repeat experiment
    for iter in range(args.iterations):

        filename = 'logs/' + test_id + str(iter) + '.csv'
        csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


        # Update number of classes and change model due to input and output shape
        cnn = trained_models.resnet18(pretrained = True)
        num_ftrs = cnn.fc.in_features
        cnn.fc = nn.Linear(num_ftrs, len(test_dataset.classes))
        cnn = torch.nn.DataParallel(cnn).cuda()

        cnn = cnn.cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                        momentum=0.9, nesterov=True, weight_decay=5e-4)

        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

        for epoch in range(args.epochs):

            xentropy_loss_avg = 0.
            correct = 0.
            total = 0.

            progress_bar = tqdm(train_loader)
            for i, (images, labels) in enumerate(progress_bar):
                progress_bar.set_description('Epoch ' + str(epoch))

                images = images.cuda()
                labels = labels.cuda()

                cnn.zero_grad()
                pred = cnn(images)

                xentropy_loss = criterion(pred, labels)
                xentropy_loss.backward()
                cnn_optimizer.step()

                xentropy_loss_avg += xentropy_loss.item()

                # Calculate running average of accuracy
                pred = torch.max(pred.data, 1)[1]
                total += labels.size(0)
                correct += (pred == labels.data).sum().item()
                accuracy = correct / total

                progress_bar.set_postfix(
                    xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
                    acc='%.3f' % accuracy)

            test_acc = test(test_loader)
            tqdm.write('test_acc: %.3f' % (test_acc))

            # scheduler.step(epoch)  # Use this line for PyTorch <1.4
            scheduler.step()     # Use this line for PyTorch >=1.4

            row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
            csv_logger.writerow(row)

        torch.save(cnn.state_dict(), 'checkpoints/' + test_id + str(iter) + '.pt')
        csv_logger.close()

        trainset, trainloader, testset, testloader = get_loaders_and_dataset(dataset, train_transform, test_transform, args.batch_size)
        targets, preds, _ = make_prediction(cnn, testset.classes, testloader)
        test_class_report = classification_report(targets, preds, target_names=testset.classes)
        test_metrics = get_metrics_from_classi_report(test_class_report)

        targets, preds, _ = make_prediction(cnn, testset.classes, trainloader)
        train_class_report = classification_report(targets, preds, target_names=testset.classes)
        train_metrics = get_metrics_from_classi_report(train_class_report)

        print(test_metrics)
        metrics = []
        metrics.extend(test_metrics)
        metrics.extend(train_metrics)
        result_df.loc[len(result_df.index)] = metrics
        result_df.to_csv('experimental_result_for_cutout.csv')


if __name__ == "__main__":
    main()
