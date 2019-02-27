## Gesture Recognition

# Features
- Dataset preparation (numpy), includes train/test and validation split, normalization etc.
- Implements Logistic Regression with gradient descent (numpy)
- Implements Logistic Regression with LBFGS (numpy)
- Implements a wrapper around sklearn LogisticRegression for comparison and testing
- Implements a Bidirectional LSTM with Keras
- Implements a Bidirectional LSTM with Tensorflow (implemented to see if unrolling up to the exact variable timestep helps improve the model)

# Installation and Package Requirements
- python 3.5
- numpy
- tensorflow
- keras
- sklearn
- matplotlib
- jupyter
- tqdm


# Run Code
- Run the notebook Models_Training_and_Evaluation
- It does the following
  - Downloads the data
  - Prepares the dataset
  - Runs each of the above models
