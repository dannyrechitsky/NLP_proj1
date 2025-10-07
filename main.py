"""
The main script for training the model and display the evaluation results.

Instructions:
---
Commonly, the main script contains following functions:
* ``train_loop``
* ``test_loop``
* Evaluation functions
* (Optional) Command line arguments or options
    * If you need explicit control over this script (e.g. learning rate, training size, etc.)
* (Optional) Any functions from ``utils.py`` that helps display results and evaluation

Eventually, this script should be run as
```
uv run main.py <ARGUMENTS> --<OPTIONS>
```

References:
---
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

from dataset import *
from model import *
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def train(dataloader,
            val_loader,
            model,
            criterion,
            optimizer,
            num_epochs=30) -> int:

    ## Initialize early stop conditions:
    # count of num epochs with increasing validation loss
    # used for early stop condition
    inc_loss_epochs = 0
    # validation loss for previous epoch, to check if loss increasing
    prev_epoch_loss = torch.inf

    # number of epochs trained if training stopped early
    epochs_trained = 0

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_input, batch_output in dataloader:
            # zero out previous gradients
            optimizer.zero_grad()

            # forward pass
            logits = model(batch_input)

            # calculate loss
            loss = criterion(logits, batch_output)
            total_loss += loss.item() * batch_input.size(0)

            # backpropagation and weight update
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataloader.dataset)
        print(f'EPOCH ------------- {epoch}')
        print(f'TRN: average loss = {avg_loss}')

        # run validation for this epoch
        avg_val_loss = validate(val_loader, model, criterion)

        # early stop condition: once validation loss increases for 3 epochs
        epochs_trained = epoch
        inc_loss_epochs = count_loss_epochs(prev_epoch_loss, avg_val_loss, inc_loss_epochs)
        if  inc_loss_epochs > 2:
            print(f'EARLY TERMINATION: validation loss ' 
                  f'increasing for 3 consecutive epochs')
            break
        else:
            print(f'--Loss incr epochs: {inc_loss_epochs}')
        prev_epoch_loss = avg_val_loss

    print(f'----END OF TRAINING---')
    print(f'----Epochs trained: {epochs_trained}')


        
        # save weights
        # torch.save(model...state_dict(), 
        #             "pickle_jar/weights_"+self.encoding)

def validate(val_loader,
            model,
            criterion):
    model.eval() # evaluation mode to avoid updating weights
    total_loss = 0.0

    with torch.no_grad(): # disable gradient calculation
        for batch_input, batch_output in val_loader:
            # forward pass
            logits = model(batch_input)

            # calculate loss
            loss = criterion(logits, batch_output)
            total_loss += loss.item() * batch_input.size(0)
        
        avg_val_loss = total_loss / len(val_loader.dataset)
        print(f'VAL: average loss = {avg_val_loss}')

        return avg_val_loss

def test(test_loader,
            model):
    model.eval() # evaluation mode to avoid updating weights

    NUM_CLASSES = 21
    # initialize accuracy
    acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES, average="macro")
    f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")

    with torch.no_grad(): # disable gradient calculation
        for batch_input, batch_output in test_loader:
            # forward pass
            logits = model(batch_input)

            # get predicted classes and update metrics
            _, predicted_labels = torch.max(logits, 1)
            acc.update(predicted_labels, batch_output)
            f1.update(predicted_labels, batch_output)

    # compute final metrics
    final_acc = acc.compute()
    final_f1 = f1.compute()

    # print results
    print("\n\n\n---TEST RESULTS---\n")
    print(f'Accuracy: {final_acc}* macroaveraged')
    print(f'F1 Score: {final_f1}* macroaveraged')





    




if __name__ == "__main__":
    INPUT_DIM = 300
    OUTPUT_DIM = 21 
    
    # compute hidden sizes for MLPs of varying depth
    hidden_sizes:list[int] = []
    DEPTH = 10
    dim : int = 256
    for i in range(DEPTH):
        hidden_sizes.append(dim)   
        if DEPTH - i > 5: 
            # reduce next hidden size
            if dim > 4:
                dim /= 2
                dim = int(dim)
            # once it reaches 4, keep next hidden size=4
                
        # expand next hidden size toward the end
        else:
            if dim < 256:
                dim *= 2
                dim = int(dim)


    mlp_model = MLP(INPUT_DIM, hidden_sizes, OUTPUT_DIM, from_pickle=True)

    class_weights_tensor = weigh_classes(
        mlp_model.data.senses, 
        mlp_model.sorted_senses
        )
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=0.001)

    features = mlp_model.features
    labels = mlp_model.senses_tensor

    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    val_set = PDTBDataset(set="validate")
    val_features = unpickle_features_val()
    val_labels = torch.tensor(
                [mlp_model.sense_map[sense] for sense in val_set.senses],
                dtype=torch.long
                )
    val_tensor_set = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_tensor_set, batch_size=256, shuffle=False)

    train(dataloader, val_loader, mlp_model, criterion, optimizer, num_epochs=30)

    test_set = PDTBDataset(set="test")
    test_features = unpickle_features_test()
    test_labels = torch.tensor(
                [mlp_model.sense_map[sense] for sense in test_set.senses],
                dtype=torch.long
                )
    test_tensor_set = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_tensor_set, batch_size=512, shuffle=False)

    test(test_loader, mlp_model)

