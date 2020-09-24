from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from hyper_qnn import HQNN, get_data
from hyper_qnn import test as m_test
from mnist_cnn import CNN
import pennylane as qml
from pennylane import numpy as np

# 0, .05, .1, .15, .2, .25, .30,0.1,0.3,0.5,0.7,
epsilons = [0.9]
num_class=[1,0]
qbit=len(num_class)
dev = qml.device("default.qubit", wires=qbit)
device = torch.device("cpu")

train_loader, test_loader=get_data(batch_size=1, classes=num_class)

# pretrained_model = "mnist_cnn_"+str(len(num_class))+".pt"
# model = CNN()
pretrained_model = "mnist_hqnn_"+str(len(num_class))+".pt"
model = HQNN()

model.load_state_dict(torch.load(pretrained_model))
# model.load_state_dict(torch.load('\parameter.pkl'))
model.eval()
m_test(model, device, test_loader)

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, device, test_loader, epsilon ):
    model.eval()
    # Accuracy counter
    correct = 0
    adv_examples = []
    count=0
    # Loop over all examples in test set
    for data, target in test_loader:
        count+=1
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_pred.item() != target.item():
        #     continue

        # Calculate the loss
        target=target.long()
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()
        #
        # # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        print(count, end=' ')
        print(final_pred.item(), end=' ')
        print(init_pred.item(), end=' ')
        print(target.item())
        if final_pred.item() == init_pred.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, 1.0, step=0.1))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()

cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()