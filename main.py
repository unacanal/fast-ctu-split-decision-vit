import torch
from vit_pytorch import ViT
from dataset import FastQTMTDataset
import torch.nn as nn
from torch.autograd import Variable
import time as t
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

batch_size = 256 # folder num # image_num = batch_size * 32
num_workers = 8 # False
epoch = 100
dir_name = t.strftime('~%Y%m%d~%H%M%S', t.localtime(t.time()))
log_train = './log/' + dir_name + '/train'
writer = SummaryWriter(log_train)

v = ViT(
    image_size=128, # 256
    patch_size=32, # 32
    num_classes=2,
    dim=512, # 1024
    depth=2,
    heads=4,
    mlp_dim=2048,
    channels=1,
    dropout=0.3,
    emb_dropout=0.3
) # small layers

# pytorch_total_params = sum(p.numel() for p in v.parameters())
# print(pytorch_total_params)
# pass

##### Hyperparams #####
# bce = nn.BCELoss()
criterion = nn.CrossEntropyLoss()
sigmoid = nn.Sigmoid()
opt = torch.optim.Adam(v.parameters(), lr=3e-4)
#opt = torch.optim.SGD(v.parameters(), lr=3e-4)
v.cuda()
# bce.cuda()
criterion.cuda()

##### Load train, val, test dataset #####
train_set = FastQTMTDataset("train")
val_set = FastQTMTDataset("val")
test_set = FastQTMTDataset("test")

count0 = 0
count1 = 0
for idx in range(len(train_set)):
    label = int(train_set[idx][1])
    if label == 0:
        count0 += 1
    else:
        count1 += 1
print("Train 0 1", count0, count1)
count0 = 0
count1 = 0
for idx in range(len(val_set)):
    label = int(val_set[idx][1])
    if label == 0:
        count0 += 1
    else:
        count1 += 1
print("Val 0 1", count0, count1)
count0 = 0
count1 = 0
for idx in range(len(test_set)):
    label = int(test_set[idx][1])
    if label == 0:
        count0 += 1
    else:
        count1 += 1
print("Test 0 1", count0, count1)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
)


def plot_confusion_matrix(cm, classes, iter, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix very prettily.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)

    # Specify the tick marks and axis text
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    # The data formatting
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Print the text of the matrix, adjusting text colour for display
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_' + str(iter) + '.png')
    plt.show()

def confusion_matrix_c(preds, labels):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(len(labels)):
        pred = torch.argmax(preds[i])
        if pred == labels[i]:
            if pred == 1:
                TP += 1
            else:
                TN += 1
        else:
            if pred == 1:
                FP += 1
            else:
                FN += 1

    return TP, FN, FP, TN

for i in range(epoch): # continue training
    ##### TRAIN #####
    for j, (img, label) in enumerate(train_loader):
        #train
        #TP, FN, FP, TN = 0, 0, 0, 0
        loss = torch.tensor(0.0).cuda()
        v.zero_grad()
        train_img = Variable(img).cuda()
        #video_label = label
        labels = Variable(label).cuda()
        preds = v(train_img)  # input:(10,3,256,256) #(1, 1000)
        loss += criterion(preds, labels)

        loss.backward()
        opt.step()

        iter = i * len(train_loader) + j
        if j % 100 == 0:
            print("[Train] {} Loss: {:.3f}".format(iter, loss.data))
            # writer
            writer.add_scalar('train_loss', loss.data, iter)
            # writer.add_images('train_image_sample', img, j)

        ##### VALIDATION #####
        if j % 1000 == 0 and j != 0:
            with torch.no_grad():
                best_acc, total_acc = 0, 0
                for k, (img, label) in enumerate(val_loader):
                    val_loss = torch.tensor(0.0).cuda()
                    val_img = img.cuda()
                    val_labels = label.cuda()
                    val_preds = v(val_img)
                    val_loss += criterion(val_preds, val_labels)

                    ##### TP, FN, FP, TN #####
                    TP_val, FN_val, FP_val, TN_val = confusion_matrix_c(val_preds, val_labels)

                    total_acc += TP_val + TN_val
                    val_accuracy = (TP_val + TN_val) / (TP_val + FN_val + FP_val + TN_val)
                    print("[Validation] {} Loss: {:.3f}, Accuracy: {:.3f}".format(k, val_loss.data, val_accuracy))

                ##### Save best model #####:
                total_acc /= len(val_set)
                if best_acc < total_acc:
                    torch.save(v.state_dict(), 'best_model+'+ str(iter) +'.pt')
                    best_acc = total_acc
                    print("===> Best model saved in epoch:", i, ", iter:", iter, ", acc:", total_acc)

                # writer
                # writer.add_scalar('test_epoch_loss', test_loss.data, j)
                writer.add_scalar('val_accuracy', total_acc, iter)
                # writer.add_scalar('test_epoch_precision', test_precision, j)
                # writer.add_scalar('test_epoch_recall', test_recall, j)
                # writer.add_scalar('test_epoch_f1score', test_f1_score, j)
                # writer.add_hparams({"test_TP": TP_test, "test_TN": TN_test, "test_FP": FP_test, "test_FN": FN_test})

train_iter = iter
##### TEST #####
v.eval()
all_preds = []
all_labels = []
total_acc = 0
with torch.no_grad():
    for iter, batch in enumerate(test_loader):
        imgs, labels = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()

        preds = v(imgs)
        loss = criterion(preds, labels)

        ##### For Confusion Marix #####
        for i in range(len(labels)):
            all_preds.append(torch.argmax(preds[i]).cpu())
            all_labels.append(labels[i].cpu())

        ##### TP, FN, FP, TN #####
        TP_test, FN_test, FP_test, TN_test = confusion_matrix_c(preds, labels)

        total_acc += TP_test + TN_test
        acc = (TP_test + TN_test) / (TP_test + FN_test + FP_test + TN_test)
        precision = TP_test / (TP_test + FP_test)
        recall = TP_test / (TP_test + FN_test)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        if iter % 100:
            print("[Test] {} Loss: {:.3f}, Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}, F1_score: {:.3f}".format(iter, loss.data, acc, precision, recall, f1_score))

    total_acc /= len(test_set)
    print(total_acc)
    writer.add_scalar('test_accuracy', total_acc, 0)
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, {"0", "1"}, train_iter-1)
writer.close()