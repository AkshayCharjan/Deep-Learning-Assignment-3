# CS6910 Assignment 3

The code snippet provided is for implementing a sequence-to-sequence (Seq2Seq) model using PyTorch. It prepares and preprocesses a dataset containing Latin-Devanagari word pairs. It defines an Encoder and Decoder class using RNN cells (LSTM, GRU, or RNN) and initializes the Seq2Seq model with these components. Finally, it performs the forward pass, encoding the source sequence and decoding it to generate the target sequence.

### Instructions to train and evaluate the model
1. Install the required libraries:
```python
!pip install pytorch_lightning

!pip install wandb
```
2. Give proper path for the dataset.
3. To train the model run train.py using the below command: 
```python
python train.py -we myname --wp myprojectname
```
Following are the supported command line arguments:

|           Name           | Default Value | Description                                                               |
| :----------------------: | :-----------: | :------------------------------------------------------------------------ |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard      |
|  `-we`, `--wandb_entity` |     myname    | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
|  `-e`, `--epochs` |     10    | Number of epochs to train neural network.. |
|  `-lr`, `--learning_rate` |     0.0001    | Learning rate used to optimize model parameters. |
|  `-do`, `--drop_out` |     0.3    | Dropout value. |
|  `-ct`, `--cell_type` |     LSTM    | choices = ["RNN", "GRU", "LSTM"]. cell type |
|  `-es`, `--input_embedding_size` |     256    | choices =[16,32,64,256]. input embedding size |
|  `-hs`, `--hidden_layer_size` |     256    | choices =[16,32,64,256]. hidden layer size |
|  `-ne`, `--number_of_encoder_layers` |     1    | choices =[1,2,3]. hidden layer size |
|  `-nd`, `--number_of_decoder_layers` |     1    | choices =[1,2,3]. hidden layer size |
|  `-bd`, `--bidirectional` |     True    | choices =[True,False]. Bidirectional|
|  `-at`, `--attention` |     True    | choices =[True,False]. Attention|

\
4. To evaluate the model, use the following command:
```python
trainer.test(model, test_dataloader)
```
### Dataset and Data Loaders
    
### Methods
The training_step method calculates the loss and accuracy of the model during training and logs them using wandb_logger. 
Similarly, the validation_step method calculates the loss and accuracy of the model during validation and logs them. 
Finally, the test_step method calculates the loss and accuracy of the model during testing.
The configure_optimizers method initializes the Adam optimizer with a specified learning rate.

The main code initializes an instance of the seqtoseq class with the specified hyperparameters and trains the model using the fit method of Trainer class provided by PyTorch Lightning. The wandb is used to log the training and validation metrics to the Weights & Biases platform. The max_epochs and devices can also be specified as command line arguments.

### sweep
The following code sets up a configuration for a parameter sweep using the wandb.sweep function. The sweep configuration is defined as a dictionary sweep_config.
```python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        
        'drop_out': {"values": [0,0.2, 0.3]},

        'input_embedding_size': {"values": [16,32,64,256]},

        'hidden_layer_size': {"values": [16,32,64,256]},


        'number_of_encoder_layers': {"values": [1, 2, 3]},

        'number_of_decoder_layers': {"values": [1, 2, 3]},


        "cell_type": {
              "values": [ "RNN", "GRU", "LSTM"]
          },
          
          "learning_rate": {
              "values": [1e-3, 1e-4]
          },
       
        "bidirectional":{
            "values":[True, False]
        },

          "epochs": {
              "values": [10, 15, 20]
          },
          "attention": {
              "values": [True, False]
          },
          
    }
}
```
The method key specifies the method used for the sweep. In this case, it is set to 'bayes', which suggests that the sweep will use Bayesian optimization to determine the best set of hyperparameters.
This search aims to maximize the validation accuracy metric of a model. The parameters section specifies various hyperparameters and their potential values for exploration during the search. These include the drop_out rate, input_embedding_size, hidden_layer_size, number_of_encoder_layers, number_of_decoder_layers, cell_type (RNN, GRU, or LSTM), learning_rate, bidirectional (True or False), epochs, and attention (True or False).

For each hyperparameter, a list of possible values to sweep over is provided using the values key. The sweep will test different combinations of hyperparameters and record the results in Weights and Biases (W&B) for analysis.

##### Use the wandb.sweep function to set up the hyperparameter sweep
sweep_id = wandb.sweep(sweep_config, project=pName)

##### Use the wandb.agent function to run the hyperparameter sweep with the given sweep_id and sweep function
wandb.agent(sweep_id, sweep)

Report Link: https://wandb.ai/cs22m008/Assignment%202%20Part%20A%20main%204/reports/CS6910-Assignment-2--Vmlldzo0MDMzMjY2




