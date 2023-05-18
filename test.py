# -*- coding: utf-8 -*-
# @Author  : Dengxun
# @Time    : 2023/5/17 8:15
# @Function:
import matplotlib.pyplot as plt
train_loss_history=[0.5,0.4,0.3,0.2]
test_loss_history1=[0.8,0.6,0.4,0.1]
test_loss_history2=[0.9,0.7,0.5,0.1]
train_acc_history=[0.1,0.5,0.8,0.9]
test_acc_history1=[0.2,0.6,0.8,0.8]
test_acc_history2=[0.1,0.2,0.6,0.9]
epoch = range(1, len(train_loss_history) + 1)
plt.figure(figsize=(8, 6))
# Plotting the loss
plt.plot(epoch, train_loss_history, label='Training Loss', color="red")
plt.plot(epoch, test_loss_history1, label='Public Test Loss', color="green")
plt.plot(epoch, test_loss_history2, label='Private Test Loss', color="blue")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
# Plotting the accuracy
# plt.savefig('loss' + ".png")
# plt.savefig('loss' + ".pdf")
plt.show()



plt.figure(figsize=(8, 6))
plt.plot(epoch, train_acc_history, label='Training Accuracy', color="red")
plt.plot(epoch, test_acc_history1, label='Public Test Accuracy', color="green")
plt.plot(epoch, test_acc_history2, label='Private Test Accuracy', color="blue")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracy')
plt.legend()
# Adjusting the layout
# plt.savefig('Accuracy' +  ".png")
# plt.savefig('Accuracy' + ".pdf")
# Display the plot
plt.show()