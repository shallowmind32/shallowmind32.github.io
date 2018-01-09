---
layout:     post
title:      Interpreting Deep Neural Networks
date:       2017-12-30
summary:    No matter how powerful our current neural networks are, we still can't use them in safety-critical domains like medicine or law because we can't understand them. How can we better interpret these deep learning models? Can we directly optimize them to be more understandable?
categories: jekyll pixyll
---

<b>tl;dr</b> <i>Interpretability is important. We should explicitly optimize for interpretable deep neural networks. Click <a href="https://github.com/dtak/tree-regularization-public">here</a> for a NumPy Autograd implementation for tree-regularization on a toy timeseries dataset.</i>

---

In the past years, deep learning has quickly become the industry and research workhorse for prediction, and rightfully so. Neural networks have time and time again been the state-of-the-art for image classification, speech recognition, text translation, and more among a growing list of difficult problems. In October, Deepmind released a more powerful version of AlphaGo that could be trained from scratch to defeat even the best human players and bots, hinting at a promising future for AI. In industry, companies like Facebook and Google have deep networks integrated at the core of their computational pipelines, thereby relying on these algorithms to process billions of bytes of data daily. Following suite, new startups like <a href="https://www.springhealth.com/">Spring</a> or <a href="https://www.babylonhealth.com/">Babylon Health</a> are adapting similar methods to disrupt the healthcare domain. Needless to say, deep learning is starting to impact our daily lives.

![grad_cam]({{ "/images/tree_regularization/grad-cam.png" | absolute_url }})
*<b>Figure 1:</b> GradCam - Creating visual explanations for decisions by using gradients of target concepts to highlight important pixels. (Selvaraju et. al. 2017).*

But deep learning is a black box. When I first heard of it, I had a really hard time grasping how it works. Years later, I am still searching for a good answer. <b>Trying to interpret modern neural networks is a difficult but extremely important problem</b>. If we are going to be relying on deep learning to make new AI, handle sensitive user data, or prescribe medication, we must be able to understand how these models think.

Lucky for us, many smart people in academia have thought about understanding deep learning. Here are some examples from recent publications:

  - <a href="https://arxiv.org/abs/1610.02391">Grad-Cam</a> (<i>Selvaraju et. al. 2017</i>): Uses the gradients from the final convolutional layer to produce a heatmap highlighting important pixels in the input image responsible for a classification.
  - <a href="https://arxiv.org/abs/1602.04938">LIME</a> (<i>Ribeiro et. al. 2016</i>): Approximates a DNN's predictions using sparse linear models where we can easily identify important features.
  - <a href="https://distill.pub/2017/feature-visualization/">Feature Visualization</a> (<i>Olah 2017</i>) Starting from an image with random noise, optimize the pixels to activate a particular neuron in a trained DNN to visualize what that neuron "has learned".
  - <a href="https://arxiv.org/abs/1712.09913">Loss Landscape</a> (<i>Li et. al. 2017</i>): Visualize non-convex loss functions that DNN try to minimize and see how architectures/parameters effect the loss landscape.

![feature_viz]({{ "/images/tree_regularization/feature-viz.png" | absolute_url }})
*<b>Figure 2:</b> Feature Vizualization - generate images via optimization to activate a specific neuron or set of neurons. (Olah 2017).*

Just from these four examples, there are different ideas of what interpreting a DNN means. Is it about isolating the effects of individual neurons? Is it about visualizing the loss landscape? Is it about feature sparsity? 

When picking a definition, I'd encourage one that <b>involves a human</b> since the point of interpretability is to help <u>us</u> explain complex behavior. It's hard to imagine an automated clinician that doesn't involve a licensed doctor somewhere down the line. 

## What is interpretability?

We should think of interpretability as <u>human simulatability</u>. A model is <u>simulatable</u> if a human can <i>take in input data together with the parameters of the model and in <b>reasonable</b> time, step through every calculation required to make a prediction</i> (<i>Lipton 2016</i>).

This is rather strict but powerful definition. Going back to a hospital ecosystem: given a simulatable model, doctors can easily check every step the model is taking against their own expert knowledge, and even reason about things like fairness and systemic bias in the data. This allows practitioners to help improve models in a positive feedback loop. 

## Decision Trees are simulatable.

It's easy to see that decision trees \(DTs) are simulatable. For example, if I'm trying to predict whether a patient is at low or high risk for a heart attack, I, as a human, can walk down each node in the tree and understand what feature(s) are being used to make a prediction.

{: .center}
![decision_tree]({{ "/images/tree_regularization/decision-tree.png" | absolute_url }})

*<b>Figure 3:</b> Decision Tree trained to classify risk of heart attack. This tree has a maximum path length of 3.*

If we could get away with using trees instead of DNNs, then we'd be done. But while we sacrifice interpretability, DNNs give us much more power than trees do. Is it possible for us to combine DTs and DNNs to get something simulatable <b>and</b> powerful?

Naively, we could try to do something like LIME where we construct a mimic DT to approximate the predictions of a trained DNN. But there are many local minima when training deep neural networks, only some of which are easy to simulate. By trying to approximate an already-trained DNN, we may end up in a minima that is difficult to simulate (produce a huge DT that is hard to walk through in reasonable time).

| Dataset | Decision Tree \(AUC) | GRU (AUC) |
|-------|----|----|
| Sepsis | 0.7017 | 0.7602 |
| EuResist | 0.7100 | 0.7516 |
| TIMIT | 0.8668 | 0.9620 |

*<b>Table 1:</b> Performance of DTs and RNNs on a variety of datasets. We note a large increase in predictive power switching from DT to RNN. (The TIMIT task is to predict stop vs non-stop characters).*


## Directly optimize for simulatability.

If we think about simulatability during optimization, then we can try to find more interpretable minima. Ideally, we would like to train a DNN to mostly behave like a decision tree but not exactly since we want to still take advantage of the non-linearities in neural nets.

Another way of stating this is that we want to <b>regularize deep neural networks using simple decision trees</b>. We call this <i>Tree Regularization</i>.

## Tree regularization.

Imagine we have a timeseries dataset with \\(N\\) sequences, each with \\(T_{n}\\) timesteps. Without restriction, we can assume binary outputs. Traditionally, if we are to train a recurrent neural network (RNN), we would use the following loss:

$$ \lambda\psi(W) + \sum_{n=1}^{N}\sum_{n=1}^{T_{n}}\mathsf{loss}(y_{nt}, \hat{y}_{nt}(x_{nt}, W)) $$

where \\(\psi\\) is a regularizer (i.e. L1 or L2), \\(\lambda\\) is a regularization strength, \\(W\\) is the set of RNN weights, \\(y_{nt}\\) is the ground truth output at a single timestep, and \\(\hat{y}\_{nt}\\) is the predicted output at a single timestep. The \\(\mathsf{loss}\\) is normally a cross-entropy loss.

Adding tree regularization requires two changes. The first part is given some RNN with weights \\(W\\) (these can be partially trained weights), we pass the training data \\(x_{1:N,1:T}\\) through the RNN to make predictions \\(\hat{y}\_{1:N,1:T}\\). We then train a decision tree using \\(x_{1:N,1:T}, \hat{y}\_{1:N,1:T}\\) to try to match the predictions of the RNN.

![tree_reg_1]({{ "/images/tree_regularization/tree-reg-1.png" | absolute_url }})
*<b>Figure 4:</b> At any point in the optimization, we can approximate a partially trained DNN with a simpler decision tree.*

At this point, we have a mimic DT. But we could have a really small or really large DT. We would like to be able to quantify the <u>size</u> of the tree.

To do so, we consider the tree's <i>average path length</i> \(APL). For a single example, the <i>path length</i> is the depth you have to travel in the tree to make a prediction. For example, consider the DT for heart attack prediction shown in Figure 3. Imagine an input \\(x\\) with an age of 70. The path length of \\(x\\) would then be 2 since 70 > 62.5. As such, the average path length is simply \\(\sum_{n=1}^{N} \mathsf{pathlength}(x_{n}, \hat{y}\_{n})\\). Another way to think about this is the cost for a human to simulate the average example. 

![tree_reg_2]({{ "/images/tree_regularization/tree-reg-2.png" | absolute_url }})
*<b>Figure 5:</b> Given a decision tree and a dataset, we can compute the average path length as the cost of simulating/interpreting the average example. By including this term in the objective, we want to encourage our DNN to produce DTs that are simple and penalize them for creating giant trees.*

We can redefine the loss function as the following:
 
$$ \lambda\sum_{n=1}^{N}\sum_{n=1}^{T_{n}} \mathsf{pathlength}(x_{nt}, \hat{y}_{nt}) + \sum_{n=1}^{N}\sum_{n=1}^{T_{n}}\mathsf{loss}(y_{nt}, \hat{y}_{nt}(x_{nt}, W)) $$

There's just one problem: <i>decision trees aren't differentiable</i>. But we really want to stick with SGD since its fast and easy to use, so we have to be a bit more creative.

What we do is we add a <b>surrogate model</b>, which is usually a multilayer perceptron \(MLP) that takes in RNN weights as input and outputs an estimate of average path length (as if we trained a decision tree). 

![tree_reg_3]({{ "/images/tree_regularization/tree-reg-3.png" | absolute_url }})

*<b>Figure 6:</b> Using the surrogate model lets us use popular gradient descent methods for training the DNN. To train the surrogate, we minimize the MSE between true and predicted APLs.* 

As we optimize the RNN/DNN, each gradient step will produce a new set of weights \\(W_{i}\\). For each \\(W_{i}\\), we can train a DT and compute the APL. Over several epochs, we can curate a large dataset to train the surrogate MLP.

The training procedure is then given a fixed surrogate, we define our regularized objective, and optimize the RNN. Then given a fixed RNN, we have built a dataset and we can optimize the MLP.

## Toy dataset.

A good sanity check with new techniques is to try them on synthetic datasets where we can accent the proposed benefits. 

Consider the following toy dataset: given points \\(x_{i}, y_{i}\\) in a unit 2-dimensional coordinate system, we define a ground-truth parabolic decision function.

$$y = 5 * (x - 0.5)^{2} + 0.4$$

We sample 500 points uniformly within the unit square \\([0,1] \times [0,1]\\). All points above the parabola are positive and all points below are negative. We add some noise by randomly flipping 10% of the points in a region near the boundary (between the two gray parabolas in Figure 7). A random 30% split was held out for testing.

For a classifier, we train a 3-layer MLP with 100 first layer nodes, 100 second layer nodes, and 10 third layer nodes. We purposefully make this model overly expressive to encourage overfitting and exaggerate the effect of regularization. 

![toy_dataset]({{ "/images/tree_regularization/toy-dataset.png" | absolute_url }})

*<b>Figure 7: </b> Toy parabola dataset - we train a deep MLP with different levels of L1, L2, and Tree regularization to examine visual differences between the final decision boundaries. The key point here is that tree regularization produces axis aligned boundaries.*

We then train a series of MLPs with varying regularizations \(L1, L2, Tree), and varying strengths \\(\lambda\\). We can approximate the learned decision function by evaluating the model against all points in the unit square and plotting a contour map. Figure 7 shows side-by-side comparisons of learned decisoin functions under the different parameters. 

As expected, with increasing regularization strength, we get simpler decision functions (reduced overfitting). More notably, the three regularization methods produce decision functions of different shapes. L1 tends to be jagged lines, L2 tends to be more bulbous, and <b>Tree regularization tends to prefer axis-aligned decision functions</b>. This makes a lot of sense given how decision trees work.

![toy_dataset]({{ "/images/tree_regularization/toy-dataset-2.png" | absolute_url }})

*<b>Figure 8: </b> Comparing performance of regularized models against APL. Here, decision tree (the yellow line) refers to a vanilla DT \(no DNN). We note a sweet spot around 1.0 to 5.0 where the tree-regularized MLP reaches higher performance at lower complexity than all other models.*

At least for this toy example, tree regularization seems to lead to better performance in high regularization \(human-simulatable) regimes. For example, tree regularization with \\(\lambda=9500.0\\) requires only 3 branches yet performs similarly to a parabolic decision function (which has a higher APL).

## Real world datasets.

Now that we have an intuitive sense of what tree regularization does, we can now move on to real world datasets (with binary-outcomes) and see how it compares to L1 and L2. Briefly, let's go over each of the datasets:

- <a href="https://mimic.physionet.org/">Sepsis</a> (<i>Johnson et. al. 2016</i>): Time-series data for over 11k septic intensive-care-unit (ICU) patients. We get at each timestep a data vector of 35 vital signs and label results (i.e. oxygen levels or heart rate) and a label of 5 binary outcomes (i.e. if a ventilation was used or mortality). 
- <a href="http://engine.euresist.org/">EuResist</a> (<i>Zazzi et. al. 2012</i>): Time-series data for 50k patients diagnosed with HIV. The structure is very similar to Sepsis but with a different set of 40 input features and 15 output features.
- <a href="https://catalog.ldc.upenn.edu/ldc93s1">TIMIT</a> (<i>Garofolo et. al. 1993</i>): recordings of 630 English speakers where each sentence contains transcriptions of 60 phonemes. We focus on distinguishing <i>stop</i> phonemes (those that stop the flow of air i.e. "b" or "g") from non-stops. The input features are continuous acoustic coefficients and derivatives.

We do the same thing as our toy dataset. Except this time we train a GRU-RNN. We again perform a set of experiments with varying regularization strengths and now, varying hidden unit sizes for the GRU. 

![real-world]({{ "/images/tree_regularization/real-world.png" | absolute_url }})

*<b>Figure 9: </b> Comparing performance of regularized models against APL for Sepsis (5/5 output dimensions), EuResist (5/15 output dimensions), and TIMIT. We see a similar (albeit more modest) effect to Figure 8 where if we constrain ourselves to small APLs, tree regularization reaches higher performance. See the full <a href="https://arxiv.org/abs/1711.06178">paper</a> for more detailed results and discussion.*

Even in noisy real-world data, we still see modest improvements in using tree regularization over L1 and L2 in small APL regions. For example, see 15-30 APL in the TIMIT plot, or 5-12 APL in Sepsis \(In-Hospital Mortality), or 18-23 APL in EuResist \(Adherence). We especially care about these low complexity "sweet spots" because this is exactly where a deep learning model is simulatable and actually usable in safety critical environments like medicine and law. 

In addition, once we've trained a tree-regularized DNN, we can train a mimic DT to see what the final tree should look like. This is a good sanity check, since we expect the the mimic DT to be simulatable and relevant to the particular problem domain. 

Below we show the mimic DT for 2 out of the 5 output dimensions of Sepsis. Since we are not doctors, we asked a clinical expert on sepsis treatment to look at these trees.

![real-world-2]({{ "/images/tree_regularization/real-world-2.png" | absolute_url }})

*<b>Figure 10: </b>Decision trees constructed to mimic the trained tree-regularized DNN for two of the five dimensions of Sepsis. Visually, we can confirm that these trees have small APL and are simulatable.*

Concerning the mechanical ventilation DT, the clinician noted that the features in the trees nodes (FiO2, RR, CO2, and paO2) and the values of the break points are medically valid, as many of these features measure breathing quality. 

For hospital mortality, he noted some apparent contradictions in our tree: some young patients with no organ failure are predicted to have high mortality rates while other young patients with organ failure are said to have low mortality rates. The clinician then began to reason about how uncaptured \(latent) variables could be influencing the decision-making process. <b>This kind of reasoning would not be possible from simple sensitivity analyses of the deep model.</b>

![real-world-3]({{ "/images/tree_regularization/real-world-3.png" | absolute_url }})

*<b>Figure 11:</b> Same as Figure 10 but for one of the output dimensions \(drug adherence) from the EuResist dataset.*

To really drive the point home, we can take a look at a mimic DT that tries to explain why a patient would have trouble adhering to a HIV drug prescription \(EuResist). Again, we consulted clinical collaborators, who confirmed that the baseline viral load and the number of prior treatment lines, which are prominent attributes in our DT, are useful predictors. Several studies (<i>Langford, Ananworanich, and Cooper 2007, Socas et. al. 2011</i>) suggest that high baseline viral loads lead to faster disease progression and hence need multiple drug cocktails. Juggling many drugs tends to make it harder for patients to adhere to a prescription.

## Interpretability is a priority.

The main takeaway here is a technique that encourages complex models to be well-approximated by human-simulatable functions without sacrificing too much on predictive performance. I think this flavor of interpretability is really powerful, and can allow domain experts to understand and approximately compute what a black-box model is doing.

The idea of AI safety is getting more and more mainstream. Many big conferences like NIPS are starting to focus more on important issues like fairness, value alignment, and interpretability in modern machine learning. And before we seriously start integrating deep learning into consumer goods and services (self-driving cars!), we really need to get a better grasp of how these models work. That means we need to develop more examples of interpretability that include human experts in the loop. No one wants another <a href="https://blogs.wsj.com/digits/2015/07/01/google-mistakenly-tags-black-people-as-gorillas-showing-limits-of-algorithms/">Google gorilla mistake</a>.

## Footnotes.

This work is to appear at AAAI 2018 as <i>Beyond Sparsity: Tree Regularization of Deep Models for Interpretability</i>
. A preprint can be found on <a href="https://arxiv.org/abs/1711.06178">ArXiv</a>. A similar version was an oral presentation at NIPS 2017 <a href="https://sites.google.com/view/timl-nips2017/submissions?authuser=0">TIML workshop</a>.

---

## FAQs

<b>How well does the surrogate MLP track the APL?</b>

Surprisingly well. In all experiments, we used a single layer MLP with 25 hidden nodes (which is a rather small network). This must suggest that there is a low dimensional representation of the weights that are predictive of APL).

![tracking]({{ "/images/tree_regularization/tracking.png" | absolute_url }})
*<b>Figure 12</b>: <i>True node count</i> refers to actually training a decision tree and computing the APL. <i>Predicted node count</i> refers to the output of the surrogate MLP.*

<b>How well does a tree-regularized model do compared to a vanilla decision tree?</b>

Each of the comparison plots above show decision tree AUCs compared with regulared-DNNs. To generate these lines, we do a grid search over different decision tree hyperparameters i.e. minimum number of samples to define a leaf, gini factor, etc. We note that in all cases, DT performance is worse than all regularization methods. This shows that tree-regularization does not just copy a DT.

<b>Is there anything similar to this in literature?</b>

Besides the related work mentioned in the beginning of this blog, model distillation/compression is probably the most similar sub-field. There the main idea to train a smaller model to mimic a deeper net. Here, we are essentially performing distillation using a DT during optimization.

<b>How are the runtimes for tree-regularization?</b>

Let's consider the TIMIT dataset \(largest dataset). An  L2-regularized GRU takes 2116 seconds per epoch. A tree-regularized GRU with 10 states takes 3977 seconds per epoch. This 3977 seconds includes the time needed to train the surrogate. In practice, we do this sparingly. For example, if we do it once every 25 epochs, we get an amortized per-epoch cost of 2191 seconds. 

<b>Are the \(final) mimic DTs stable over multiple runs?</b>

If the tree regularization is strong (high \\(\lambda\\)), the final DTs are stable across different runs (differing by a couple nodes at most). See paper for more details.

<b>How faithful are the DTs to deep model predictions?</b>

In other words, this question is asking if the predictions of the DTs created during training match closely to the DNN predictions. If they don't, then we aren't really regularizing our model very effectively. However, we do not expect this to be an exact match. 

![fidelity]({{ "/images/tree_regularization/fidelity.png" | absolute_url }})

In the table above, we measure <b>fidelity</b> (<i>Craven and Shavlik 1996</i>), which is the percentage of test examples on which the prediction made by the DT agrees with the DNN. It follows that the DTs are faithful.

---

## A residual GRU-HMM model.

(This section talks about a new model designed for interpretability.)

A <b>hidden markov model</b> \(HMM) is like a stochastic RNN. It models some latent variable sequence \\([z_{1}, ..., z_{T}]\\) where each latent variable is one of \\(K\\) discrete states: \\(z_{t} \in \{1, \cdots, K \}\\). The state sequence is used to generate the data \\(x_{t}\\) and outputs \\(y_{t}\\) observed at each timestep. Notably, it includes a transition matrix \\(A\\) where \\(A_{ij}=\mathsf{Pr}\(z_{t}=i \| z_{t-1}=j\)\\) and some emission parameters that generate data. HMMs are generally considered to be a more interpretable model since the \\(K\\) latent variables that cluster the data are usually semantically meaningful.

We define a <b>GRU-HMM</b> as a GRU that models the residual errors when predicting a binary target using the HMM latent states (in other words, only use the GRU when the HMM is insufficient in capturing the data). By nature of being a residual model, we can penalize the complexity of the GRU output node alone using tree regularization, leaving the HMM unconstrained.

![gruhmm]({{ "/images/tree_regularization/gruhmm.png" | absolute_url }})

*<b>Figure 13</b>: Diagram of a GRU-HMM. Here \\(x_{t}\\) represents an input data at timestep \\(t\\); \\(s\_{t}\\) respresents a latent state at timestep \\(t\\); \\(r\_{t}\\), \\(h\_{t}\\), \\(\tilde{h}\_{t}\\), \\(z\_{t}\\) represent GRU variables. The final sigmoid (next to orange triangle) is cast on top of the sum of the HMM state and the GRU hidden state multipled by some set of weights. The orange triangle indicates the output used in surrogate training for tree regularization.*

Overall, deep residual models perform about 1% better than GRU-only models with roughly the same number of model parameters. See paper supplement for more details.

![gruhmm-2]({{ "/images/tree_regularization/gruhmm-2.png" | absolute_url }})

*<b>Figure 14</b>: Like before, we can make plots and visualize the mimic DT for these residual models. While we see similar "sweet spot" behavior, we note that the resulting trees have distinct structure, suggesting that the GRU behaves differently in this residual setting.*

---

Thanks for reading!

\- M 