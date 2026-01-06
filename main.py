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
import itertools
import torch.optim as optim

def parse_args(input_list=None):
    parser = argparse.ArgumentParser()

    ### ADD DATA PATHS ###

    # add train data path
    parser.add_argument("train_path", 
                        help="Path to training set")
    
    # add validation data path
    parser.add_argument("val_path", 
                        help="Path to validation set")
    
    # add test data path
    parser.add_argument("test_path", 
                        help="Path to test set")
    

    ### ADD HYPERPARAMETERS ###
    
    # add model type
    parser.add_argument("--model",
                        default="MLP",
                        help="set model type: LR, MLP or CNN; default MLP")

    # add num epochs
    parser.add_argument("--num_epochs",
                        type=int, 
                        default=10,
                        help="set number of epochs, default 10")
    
    # add embedding type
    parser.add_argument("--embedding",
                        default='glove',
                        help="set how embeddings are encoded, " \
                        "random or glove (dense) or sparse. Default glove")
    
    # add learning rate
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="set learning rate, default 0.001")
    
    # add hidden depth
    parser.add_argument("--depth",
                        type=int,
                        default=50,
                        help="set number of hidden layers, default 50")
    
    # load from pickled model
    parser.add_argument("--from_pickle",
                        action="store_true",
                        help="load model from pickled version, else train fresh")
    
    # add max sentence length
    parser.add_argument("--max_length",
                        type=int,
                        default=50,
                        help="set max number of tokens in a sentence, default 50")
    
    
    return parser.parse_args(input_list)


def train(cli_args,
          dataloader,
          val_loader,
          model,
          criterion,
          optimizer):

    # set num_epochs from CLI args
    num_epochs = cli_args.num_epochs
    
    ## Initialize early stop conditions:
    # count of num epochs with increasing validation loss
    # used for early stop condition
    inc_loss_epochs = 0
    # validation loss for previous epoch, to check if loss increasing
    prev_epoch_loss = torch.inf

    # number of epochs trained if training stopped early
    epochs_trained = 0

    print("TRAINING LOOP IN PROGRESS...")
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

    if hasattr(model, 'output_layer'):
        # Matches MLP class
        num_classes = model.output_layer.out_features
    elif hasattr(model, 'fc'):
        # Matches CNN class
        num_classes = model.fc.out_features
    else:
        # Matches LR class
        num_classes = model.weights.out_features

    # initialize accuracy
    acc = Accuracy(task="multiclass", num_classes=num_classes, average="macro")
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

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

    return final_acc

def run_model(
            cli_args, 
            sentence_type = "concat",
            weight_decay = 5e-3,
            dropout = 0.7,
            train_batch_size = 64,
            hidden_shape = 'rectangle',
            hidden_width = 128,
            is_grid_search = False,
            # set above to True in grid_search function to search over hyper params
            gs_hyp_params=None
            # set above to dict {"encoding":x, "depth":y} in grid_search function
            ):
    
    train_data = PDTBDataset(cli_args, set="train")
    num_classes = len(set(train_data.senses))

    ## Set CLI args
    model_used = cli_args.model
    from_pickle = cli_args.from_pickle
    learn_rate = cli_args.lr

    # grid search?
    if is_grid_search:
        # set to current hyperparameter being tested
        encoding = gs_hyp_params["encoding"]
        depth = gs_hyp_params["depth"]
    else:
        # set as CLI args
        encoding = cli_args.embedding
        depth = cli_args.depth

    # TODO: change these hyperparameters into docstring
    # encoding = "glove"
    # sentence_type = "concat"

    # weight_decay = 5e-3
    # dropout = 0.7
    # learn_rate = 0.001
    # train_batch_size = 64
    # depth = 8
    # hidden_shape = 'rectangle' #or "pyramid"
    # # size of each hidden layer (only for rectangle shape)
    # hidden_width = 64 

    # compute hidden sizes for MLPs of varying depth
    hidden_sizes:list[int] = []
    
    
    dim : int = 256
    
    # define shape of MLP
    if hidden_shape == "pyramid":
        for i in range(depth):
            hidden_sizes.append(dim)   
            if depth - i > 5: 
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

    elif hidden_shape == "rectangle":
        hidden_sizes = [hidden_width for i in range(depth)]
    
    else:
        raise ValueError(f"hidden shape {hidden_shape} is invalid!/n"
                         f"It can be 'rectangle' or 'pyramid'.")


    # ----------------------------------------------------#

    # Instantiate model
    if model_used=="MLP":
        model = MLP(cli_args, 
                    hidden_sizes=hidden_sizes, 
                    output_dim=num_classes, 
                    from_pickle=from_pickle,
                    encoding=encoding, 
                    sentence_type=sentence_type, 
                    dropout=dropout)
    
    elif model_used=="LR":
        model = LogisticRegression(cli_args,
                                   output_dim=num_classes)

    else: # model_used == "CNN":
        model = CNN(cli_args,
                    batch_size=train_batch_size,
                    embed_dim=300,
                    output_dim=num_classes,
                    max_sent_len=50,
                    encoding=encoding,
                    sentence_type="concat",
                    from_pickle=from_pickle
        )

    # For model attributes message printed later
    if isinstance(model, MLP):
        model_type="MLP"
    elif isinstance(model, CNN):
        model_type="CNN"
    else:
        model_type="LR"

    
    # balancing classes
    class_weights_tensor = weigh_classes(
        model.data.senses, 
        model.sorted_senses
        )
    
    # initialize loss function
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)

    features = model.features
    labels = model.senses_tensor

    dataset = TensorDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

    val_set = PDTBDataset(cli_args, set="validate")

    # unpickle OR featurize from scratch
    if cli_args.from_pickle and cli_args.embedding != 'sparse':
        val_features = unpickle_features_val(encoding=cli_args.embedding,
                                             sentence_type=sentence_type)
    else:
        val_features = val_set.featurize(cli_args, 
                                         sentence_type=sentence_type, 
                                         external_vocab=model.data_vocab)

    # val labels
    val_labels = torch.tensor(
                [model.sense_map[sense] for sense in val_set.senses],
                dtype=torch.long
                )
    val_tensor_set = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_tensor_set, batch_size=256, shuffle=False)

    if model_used != "LR": 
        print(f'\n\n\n---{model_type} model initialized with {depth} hidden layers---')
        print(f'Hidden sizes: {hidden_sizes}')
    
    print(f'Embedding type: {model.encoding}')
    if model_type == "MLP":
        print(f'Sentence structure: {model.sentence_type}')
        print(f'Regularization: Dropout rate: {model.dropout}')
        print(f'Regularization: Weight decay: {weight_decay}')
    print(("------------------"))



    # TRAINING
    train(cli_args, dataloader, val_loader, model, criterion, optimizer)


    # TESTING
    test_set = PDTBDataset(cli_args, set="test")

    # unpickle OR featurize from scratch
    if cli_args.from_pickle:
        test_features = unpickle_features_test(encoding=cli_args.embedding, 
                                               sentence_type=sentence_type)
    else:
        test_features = test_set.featurize(cli_args, 
                                           sentence_type=sentence_type,
                                           external_vocab=model.data_vocab)

    # test labels    
    test_labels = torch.tensor(
                [model.sense_map[sense] for sense in test_set.senses],
                dtype=torch.long
                )
    test_tensor_set = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_tensor_set, batch_size=512, shuffle=False)

    acc = test(test_loader, model)

    return acc

def grid_search(cli_args):
    hparams = {
        "encoding":["glove", "random"], 
        "depth":[50, 100, 200]
            }
    
    # create iterator of hyperparameter combinations
    keys = hparams.keys()
    val_combos = itertools.product(*hparams.values())
    
    # keep track of max accuracy and best hyperparameters
    max_param_group = {}
    max_acc = 0.0
    
    for combo in val_combos:
        # reconstruct map of hyperparameter group 
        param_group = dict(zip(keys, combo))
        
        # set variables for readability
        encoding = param_group["encoding"]
        depth = param_group["depth"]

        hyperparams = {"encoding":encoding, "depth":depth}

        # run model
        acc = run_model(cli_args,
                        is_grid_search=True,
                        gs_hyp_params=hyperparams)
        
        # update max accuracy and best hyperparameters
        if acc > max_acc:
            max_acc = acc
            max_param_group = param_group
    
    print(f'/n/n/n--------Gridsearch Complete!--------')
    print(f'Best accuracy:        {max_acc}')
    print(f'Best hyperparameters: {max_param_group}')    

    return max_param_group, max_acc


def main(args=None):
    if args is None:
        args = parse_args()

    print("Initializing...")
    print(f"Train path     : {args.train_path}")
    print(f"Validation path: {args.val_path}")
    print(f"Test path      : {args.test_path}\n")


    
    # grid_search(cli_args)  ## UNCOMMENT to run grid search instead
    print("running best model")
    run_model(args)

if __name__ == "__main__":
    main()
