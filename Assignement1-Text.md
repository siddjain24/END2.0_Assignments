1.  What is a neural network neuron?

> A neuron is a fundamental unit of our brain. Neuron, w.r.t. our brain
> consists of a small "memory storage" or a "signal", as well as a
> "small computational unit". When we refer to neurons in NNs we only
> consider a small "memory storage" or a "signal" and keep the
> computation unit outside. This computation unit consists of two
> elements, a **weight, **and an **activation **function. Each neuron in
> both cases has input connections. Input connections to a brain neuron
> are called a dendrite and output connection is called an axon. Both
> are called just input and output weights in NNs. 
>
> In our brain, we have different kinds of neurons doing different
> things (the brain's activation function) with the information coming
> in. In the case of NNs, we generally have a single activation like
> tanh/sigmoid/ReLU, etc. 

Let there be n inputs to the neuron, the output can be represented as:

![](media/image1.png){width="3.25in" height="0.75in"}

where, b = bias of the neuron

![](media/image2.JPG){width="4.825in" height="3.851535433070866in"}

1.  What is the use of the learning rate?

> Learning rate is a hyper-parameter that controls how much we are
> adjusting the weights of our network with respect the loss
> gradient. The lower the value, the slower we travel along the downward
> slope.
>
> Learning rate is used to scale the magnitude of parameter updates
> during gradient descent. The choice of the value for learning rate can
> impact two things: 1) how fast the algorithm learns and 2) whether the
> cost function is minimized or not. Below Fig shows the variation in
> cost function with a number of iterations/epochs for different
> learning rates. 
>
> ![](media/image3.JPG){width="5.366666666666666in"
> height="4.3606397637795276in"}
>
> It can be seen that for an optimal value of the learning rate, the
> cost function value is minimized in a few iterations (smaller time).
> This is represented by the blue line in the figure. If the learning
> rate used is lower than the optimal value, the number of
> iterations/epochs required to minimize the cost function is high
> (takes longer time). This is represented by the green line in the
> figure. If the learning rate is high, the cost function could saturate
> at a value higher than the minimum value. This is represented by the
> red line in the figure. If the learning rate selected is very high,
> the cost function could continue to increase with iterations/epochs.
> An optimal learning rate is not easy to find for a given problem.

1.  How are weights initialized?

> The aim of weight initialization is to prevent layer activation
> outputs from exploding or vanishing during the course of a forward
> pass through a deep neural network. If either occurs, loss gradients
> will either be too large or too small to flow backwards beneficially,
> and the network will take longer to converge, if it is even able to do
> so at all.
>
> The most used weight initialization techniques are described as
> follows

  **Initialization method**                                  **Explaination**                                                                                                                                                                                                                  **Pros.**                                                               **Cons.**
  ---------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ----------------------------------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **All-zeros initialization and Constant initialization**   This method sets all weights to zeros (respectively to constant). Also, all activations in all neurons are the same, and therefore all calculations are the same, making which makes the concerned model a linear model           Simplicity                                                              Symmetry problem leading neurons to learn the same features
  **Random initialization**                                  This technique improves the symmetry-breaking process and provides much greater precision. The weights are initialized very near to zero and randomly. This method prevents from learning the same feature for input parameters   Improves the symmetry-breaking process                                  - A saturation may occur leading to a vanishing gradient - The slope or gradient is small, which can cause the gradient descent to be slow
  **LeCun initialization : normalize variance**              LeCun initialization aims to prevent the vanishing or explosion of the gradients during the backpropagation by solving the growing variance with the number of inputs and by setting constant variance.                           Solves growing variance and gradient problems                           - Not useful in constant-width networks - Takes into account the forward propagation of the input signal - This method is not useful when the activation function is non-differentiable
  **Xavier initialization (Glorot initialization)**          Xavier proposed a more straightforward method, where the weights such as the variance of the activations are the same across every layer. This will prevent the gradient from exploding or vanishing.                             Decreases the probability of the gradient vanishing/exploding problem   - This method is not useful when the activation function is non-differentiable - Dying neuron problem during the training
  **He initialization (Kaiming initialization)**             This initialization preserves the non-linearity of activation functions such as ReLU activations. Using the He method, we can reduce or magnify the magnitudes of inputs exponentially                                            Solves dying neuron problems                                            - This method is not useful for layers with differentiable activation function such as ReLU or LeakyReLU

1.  What is "loss" in a neural network?

> The Loss Function is one of the important components of Neural
> Networks. Loss is nothing but a prediction error of Neural Net. And
> the method to calculate the loss is called Loss Function. In simple
> words, the Loss is used to calculate the gradients. And gradients are
> used to update the weights of the Neural Net. This is how a Neural Net
> is trained.
>
> Few of the essential loss functions, which could be used for most of
> the objectives.

-   **Mean Squared Error (MSE)** - **MSE** loss is used for
    regression tasks. As the name suggests, this loss is calculated by
    taking the mean of squared differences between actual(target) and
    predicted values.

-   **Binary Crossentropy (BCE) – BCE** loss is used for the binary
    classification tasks. If you are using BCE loss function, you just
    need one output node to classify the data into two classes. The
    output value should be passed through a *sigmoid *activation
    function and the range of output is (0 – 1).

-   **Categorical Crossentropy (CC)** – When we have a multi-class
    classification task, one of the loss function you can go ahead is
    this one. If you are using CCE loss function, there must be the same
    number of output nodes as the classes. And the final layer output
    should be passed through a *softmax* activation so that each node
    output a probability value between (0–1).

-   **Sparse Categorical Crossentropy (SCC)** – This loss function is
    almost similar to CCE except for one change. When we are
    using SCCE loss function, you do not need to one hot encode the
    target vector. If the target image is of a cat, you simply pass 0,
    otherwise 1. Basically, whichever the class is you just pass the
    index of that class.

1.  What is the "chain rule" in gradient flow?

    ![](media/image4.png){width="1.6994750656167978in"
    height="0.8807502187226597in"}If a variable z depends on the
    variable y, which itself depends on the variable x, so that y and z
    are dependent variables, then z, via the intermediate variable of y,
    depends on x as well. This is called chain rule and mathematically
    it can be expressed as -

    Let’s Look at an example

    ![](media/image5.png){width="6.267361111111111in" height="3.625in"}

    The figure shows a node with inputs (x,y) and output z=h(x,y), where
    h represents the function performed at the node. Let's assume that
    the gradient ∂L/∂z is known. Using the Chain Rule of
    Differentiation, the gradients ∂L/∂x and ∂L/∂y can be computed as:

2.  ![](media/image6.png){width="1.453125546806649in"
    height="1.3474431321084865in"}

    This rule can be used to compute the effect of the node on the
    gradients flowing back from right to left.

    The figure shows a node with inputs (x,y) and output z=h(x,y), where
    h represents the function performed at the node. Let's assume that
    the gradient ∂L/∂z is known. Using the Chain Rule of
    Differentiation, the gradients ∂L/∂x and ∂L/∂y can be computed as:

    ![](media/image6.png){width="1.453125546806649in"
    height="1.3474431321084865in"}

    This rule can be used to compute the effect of the node on the
    gradients flowing back from right to left
