# My CNN

Implement a small CNN using PyTorch and visualize its feature maps during training.

## Instructions

Complete the code in `src/run.py`:
- Implement the model according to the specified instructions:
    - CNN with 3 convolutional layers, each followed by ReLU and MaxPooling (expect for the last layer): 
        - Conv1: in_channels=1, out_channels=8, kernel_size=3, padding=...
        - Conv2: in_channels=8, out_channels=8, kernel_size=3, padding=...
        - Conv3: in_channels=8, out_channels=8, kernel_size=3, padding=...
    - Avergae pooling before the final fully connected layer: 
        - FC: in_features=8, out_features=10
- Define the loss function and optimizer.
    - Use CrossEntropyLoss for classification.
    - Use Adam optimizer with a learning rate of 1e-3.
- Implement the training and evaluation loops.
- (Given) Every 3 epochs (and at epoch 1), visualize the feature maps of selected channels from each convolutional layer using the provided `visualize_feature_maps` function in `src/visu.py`.

- Train the model for 30 epochs on the MNIST dataset: `python -m src.run`

- Try to add Dropout Layers after each convolutional layers, batch norm layers, or change the learning rate and see how it affects the training and the feature maps.

- Do not forget to commit and push your changes regularly!