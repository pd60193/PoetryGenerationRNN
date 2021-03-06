# PoetryGenerationRNN
This project genrates poetry using a recurrent neural network with unsupervised machine learning on 1500 lines of Robert Frost's poems. The aim of the project is to generate a few lines of similar poetry as Robert Frost.

## Dataset

The **util/robert_frost.txt** contains poetry lines from Robert Frost's Poems. We train our recurrent model to learn this data and understand the pattern. First, we create a word to index dictionary which contatins a mapping between every unique word and an integer index. Following this we create an input vocabulary vector(V) for each line **l** in the poetry. Line **l** may contain m words, so the vector V would be of length m containing the indexes of the words from the word-to-index dictionary.In mathematical terms, V[i] = j, where i is the position of the word in the dictionary and j is the position of the word in the given line.

## Code

The file **PoetryGeneration/poetry_generation.py** contains the code for training on the Robert Frost dataset. The code has been written in Python using theano. It performs a stochastic gradient descent on a recurrent neural network propgating the error backwards at each layer.

- We use a **Word Embedding Layer** where we transform input for each line from V-sized vector to a D-sized vector where D<<<V. This transformation is brought by the **Word Embedding Matrix (We)** of size V X D. The weights of this matrix is trained over time. This reduces the amount of memory required and also makes it computationally more efficient.

- We use a Simple Recurrent unit with one layer and 500 hidden units. The input to this layer is the Word Vector of size D for every word and output is a single word which follows the input sequence. The input and output are as follows

    * **Input is a Vector of size D for each word**: This is the word vector signifying the features of a particular word by D weights. It has been created by transformation of vocabulary vector of size (V). The vocabulary vector V contains the word indexes from the word-to-index dictionary in the order in which they appear in that line. In addition it contains a "0" as the first element signifying the start of a line.
    Hence, "I love you" would generate a vector V = [0,4,2,3] if the vocabulary dictionary is defined as D = {"love":2, "you":3, "I":4, "caught":5,  "sugar":6, "but":7}.

    * **Output is a word which follows the input sequence** : This is the output vector which contains the same number of elements as the input vocabulary vector. The last value in the vector is the prediction. If the last value is "1", it signifies "End of Line".
    For the above example, the output could be V = [3,1,2,6] signifying the sentence "I love you but". Hence, the word "but" is the output word.
    The ouput could also be V = [4,2,3,1] signifying "I love you[EOL]" (EOL = end of line). Here "EOL" is the output word


## Utility Code

The Utility Code is in the util_parity.py file contains the following functions -:

- error_rate : takes out the error rate during training.
- get_robert_frost : gets the robert frost data and creates the vocabulary dictionary.
- init_filter : Initializes the weights along with bias
- remove_punctuation : Removes punctuations from words and makes it easier to deal with.

## Output

After running for 2000 iterations and a learning rate of 0.0001, I got an classification rate of 0.70. 

it also generated a 4 line poem and it was fairly decent.
