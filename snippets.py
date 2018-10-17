total_train_err += torch.sum(labels.argmax(dim=1) != outputs.argmax(dim=1)).item()
labels = torch.Tensor(label_binarize(labels, classes=range(0, 26))).to(device)

