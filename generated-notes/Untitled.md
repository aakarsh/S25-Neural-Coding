
# Neural Coding - Lecture 6: Correlations

## Slide 1


---
title: Neural Coding - 6. Correlations
author: Anna Levina
date: May 21, 2025
---

**Notes:**
This is the title slide for the sixth lecture in the Neural Coding series, focusing on correlations. The lecture is given by Anna Levina.

## Slide 2



---
### Previous Setting
---

**(Image: Top left - A graph showing a bell curve representing a tuning curve f(s) over a stimulus s. Points on the curve indicate responses. Bottom left - A sine wave labeled "Amplitude" vs "Time (ms)", possibly illustrating stimulus or response over time. Right - A graph showing probability distributions p(r|s) for different responses r, with stimulus s on the x-axis and response p(r|s) on the y-axis. Specific distributions for r=4, r=5, and r=6 are highlighted.)**

**Encoding:**
* Stimulus: s
* Tuning curve: f(s)
* Probability of response given stimulus: $p(r|s)$

**Decoding:**
* Error: $(s-\hat{s})^{2}$ (squared difference between actual stimulus and estimated stimulus)

**Recap of Single Neuron Response:**
* We previously considered the response 'r' of a single neuron to a stimulus 's'. [cite: 3783]
* This response was modeled as a noisy function of 's'. [cite: 3783]
* It included a tuning curve $f(s)$. [cite: 3783]
* Trial-to-trial variability was described by $p(r|s)$. [cite: 3783]

**Question for Many Neurons:**
* In the previously described coding setup, what shall we do with many neurons? [cite: 3784]
* **Option 1: Summing for Noise Reduction:**
    * Have a large number of neurons coding for the same stimulus. [cite: 3785]
    * Sum their responses to decrease noise in spike generation. [cite: 3785]
    * This works straightforwardly if the neurons are independent of each other (depending only on the input). [cite: 3786]

**Notes:**
This slide reviews the previous lecture's concepts on how a single neuron encodes a stimulus, including the tuning curve and probabilistic nature of the response. It then poses the question of how to handle responses from multiple neurons, suggesting that summing independent responses can reduce noise.

## Slide 3



---
### General Setup, Correlated Populations
---

**Problem with Independence Assumption:**
* Real neurons are not independent. [cite: 3787]
* What happens when the variability of different neurons is correlated? [cite: 3787]

**Simplified Model for Correlated Response:**
* We assume that the response 'r' follows a multivariate normal distribution:
    $r \sim \mathcal{N}(f(s), \Sigma(s))$
    * Where $f(s)$ is the mean response (tuning curve as a vector for the population).
    * $\Sigma(s)$ is the covariance matrix of the response $r(s)$. [cite: 3788]

* **Covariance Matrix $\Sigma(s)$:**
    * $\Sigma(s) = cov[r(s)] = E[(r-E[r])(r-E[r])^{T}] = E[rr^{T}] - E[r]E[r]^{T}$ [cite: 3788]
        * This formula defines the covariance matrix as the expected value of the outer product of the deviation of 'r' from its mean. It can also be calculated as the expected value of $rr^T$ minus the outer product of the expected values of 'r'.

* **Individual Entries of the Covariance Matrix:**
    * For any two neurons 'i' and 'j', the covariance is:
        $\Sigma_{ij} = \Sigma_{ji} = E[(r_{i}-E[r_{i}])(r_{j}-E[r_{j}])]$ [cite: 3788]
        * This measures how the responses of neuron 'i' and neuron 'j' vary together with respect to their individual mean responses.

**Notes:**
This slide introduces the concept of correlated variability among neurons, acknowledging that the independence assumption is often not met in real neural populations. It presents a simplified model where the population response is drawn from a multivariate normal distribution, characterized by a mean response vector and a covariance matrix. The formulas for the covariance matrix and its individual entries are provided.

## Slide 4



---
### Covariance Matrix: Properties
---

* **Transformation Property:**
    * If the input is a $1 \times d$-dimensional random vector **x** (where 'd' was 'r' in the slide, corrected for clarity), **A** is an $n \times m$ matrix, and **a** is a $1 \times m$ constant vector, then:
        $cov(A\mathbf{x}+a) = A \cdot cov(\mathbf{x}) \cdot A^{T} = A \Sigma(\mathbf{x}) A^{T}$ [cite: 3789]
        * This property describes how the covariance matrix transforms when the random vector undergoes a linear transformation.

* **Positive Semi-Definite Property:**
    * $\Sigma(\mathbf{x})$ is positive semi-definite. [cite: 3789]
    * Any symmetric positive semi-definite matrix is a covariance matrix for some random variable. [cite: 3789]
    * A matrix **M** is positive semi-definite if and only if for all non-zero vectors **x** in $\mathbb{R}^{n}$:
        $\mathbf{x}^{T}M\mathbf{x} \ge 0$ [cite: 3789]

* **Reminder on Positive Semi-Definite Matrices:**
    * Positive semi-definite matrices have all eigenvalues greater than or equal to zero. [cite: 3789]

**Notes:**
This slide details important mathematical properties of covariance matrices. It explains how covariance matrices behave under linear transformations and introduces the concept of positive semi-definiteness, a fundamental characteristic of covariance matrices, relating it to eigenvalues.

## Slide 5



---
### Correlated Variability and Accuracy of a Population Code
---

**Scenario:**
* N neurons code for the same thing. [cite: 3790]
* They have the same tuning curve $f(s)$. [cite: 3790]
* They have the same noise in spike generation. [cite: 3790]

**Simple Population Code (Average of Rates):**
* $\hat{r}_{N} = \frac{1}{N} \sum r_{i}$ [cite: 3791]
* We want $\hat{r}_{N} \rightarrow r = f(s)$ (the average rate should approach the true underlying rate). [cite: 3791]
* And $var[\hat{r}_{N}] \rightarrow 0$ (the variance of the average rate should approach zero as N increases). [cite: 3792]

**Variance if Neurons are Independent:**
* $var[\frac{1}{N}\sum_{i=1}^{N}r_{i}] = \frac{1}{N^{2}} var[\sum_{i=1}^{N}r_{i}] = \frac{1}{N^{2}} N \cdot var[r_{i}] = \frac{var[r_{i}]}{N}$ [cite: 3792]
    * This shows that if neurons are independent, the variance of the population average decreases by a factor of N.

**General Variance Formula (Including Covariance):**
* $var(\sum_{i=1}^{N}X_{i}) = \sum_{i=1}^{N}var(X_{i}) + \sum_{i \ne j}cov(X_{i},X_{j})$ [cite: 3792]
    * This formula is key when neurons are NOT independent.

**New Setting: Include Correlations:**
* Neurons have rates $r_i$. [cite: 3793]
* Individual mean rates $f_i$. [cite: 3793]
* Same variance $\sigma^2$. [cite: 3793]
* But, the covariance term is:
    $\langle(r_{i}-f_{i})(r_{j}-f_{j})\rangle = \sigma^{2}[\delta_{ij} + c(1-\delta_{ij})]$ [cite: 3793]
    * Where $\delta_{ij} = 1$ if $i=j$ (variance term) and $0$ otherwise (covariance term).
    * This introduces a correlation coefficient 'c'.

* **Correlation Coefficient 'c':**
    * $0 \le c < 1$ [cite: 3794]

**Notes:**
This slide investigates how correlated variability affects the accuracy of a population code where multiple neurons encode the same stimulus. It starts by showing how variance decreases with the number of neurons if they are independent. Then, it introduces a model for correlated neurons, defined by a constant variance $\sigma^2$ and a pairwise correlation coefficient 'c'.

## Slide 6



---
### Proof: Variance of Sign-Alternating Population Code
---

**(Note: The slide title refers to a "sign-alternating population code", but the derivation here appears to be for the variance of the average of N correlated neurons as defined on the previous slide, not specifically a sign-alternating code unless $\tilde{R}$ implies that. Assuming it's for the average $\hat{r}_N$ as defined before, but perhaps $\tilde{R}$ refers to some weighted average not fully specified on the previous slide. The derivation presented here is for $var[\tilde{R}] = var[\frac{1}{N}\sum_{i} (-1)^i r_i]$ if N is even and neurons have specific correlation structure.)**

*Using linearity and covariance structure:*
Let $\tilde{R}$ be a population code (e.g. the average, or a differently weighted sum). The variance is generally:
$var[\tilde{R}] = \frac{1}{N^{2}}\sum_{i,j} w_i w_j cov(r_{i},r_{j})$
If we consider a specific code where $w_i = (-1)^i$ (a sign-alternating sum for an even N, divided by N):
$var[\tilde{R}] = \frac{1}{N^{2}}\sum_{i,j}(-1)^{i}(-1)^{j}cov(r_{i},r_{j})$ [cite: 3795]

*Split into diagonal and off-diagonal terms:*
Using the covariance structure from the previous slide: $cov(r_i, r_j) = \sigma^2$ for $i=j$ (diagonal) and $c\sigma^2$ for $i \ne j$ (off-diagonal).
$var[\tilde{R}] = \frac{1}{N^{2}} \left( \sum_{i=i} (-1)^i (-1)^i cov(r_i, r_i) + \sum_{i \ne j}(-1)^{i}(-1)^{j}cov(r_{i},r_{j}) \right)$
$var[\tilde{R}] = \frac{1}{N^{2}} \left( N\sigma^{2} + \sum_{i \ne j}(-1)^{i}(-1)^{j}c\sigma^{2} \right)$ [cite: 3795]

*If N is even, then $\sum_{i=1}^{N}(-1)^{i}=0$. This implies:*
$(\sum_{i=1}^{N}(-1)^{i})^{2} = 0$
$0 = \sum_{i=1}^{N}(-1)^{i}(-1)^{i} + \sum_{i \ne j}(-1)^{i}(-1)^{j}$
$0 = \sum_{i=1}^{N}(1) + \sum_{i \ne j}(-1)^{i}(-1)^{j}$
$0 = N + \sum_{i \ne j}(-1)^{i}(-1)^{j}$
$\Rightarrow \sum_{i \ne j}(-1)^{i}(-1)^{j} = -N$ [cite: 3795]

*Substitute this back into the variance equation:*
$var[\tilde{R}] = \frac{1}{N^{2}}(N\sigma^{2} - N c\sigma^{2})$
$var[\tilde{R}] = \frac{N\sigma^{2}(1-c)}{N^{2}}$
$var[\tilde{R}] = \frac{\sigma^{2}}{N}(1-c)$ [cite: 3795]

**Notes:**
This slide provides a proof for the variance of a specific type of population code, a "sign-alternating" sum, under the assumption that N (number of neurons) is even and the neurons exhibit the pairwise correlation structure defined previously. The key result is that for this specific code, the variance is $\frac{\sigma^{2}}{N}(1-c)$. This shows that positive correlations ($c>0$) increase the variance compared to the independent case, but this specific coding scheme leverages the correlation structure to achieve a different outcome than a simple average. *Self-correction: The slide explicitly shows $var[\tilde{R}] = \frac{1}{N^{2}}(N\sigma^{2} + \sum_{i \ne j}(-1)^{i}(-1)^{j}c\sigma^{2})$ and then $var[\tilde{R}]=\frac{\sigma^{2}}{N}(1-c)$. This means that the specific values of $(-1)^i (-1)^j$ combined with the sum being $-N$ leads to the final result. This is a specific result for a sign-alternating sum where weights are $w_i = (-1)^i/N$. A simple average (weights $w_i=1/N$) would lead to $var[\hat{r}_N] = \frac{\sigma^2}{N^2} (N + N(N-1)c) = \frac{\sigma^2}{N}(1+(N-1)c)$, which is different.*

## Slide 7



---
### Fisher Information for Gaussian Population Codes
---

**Reference:**
* We follow Abbott and Dayan ("The effect of correlated variability on the accuracy of a population code"). [cite: 3796]

**Neuron Response Model:**
* Each neuron's response $r_i$ is modeled as:
    $r_{i} = f_{i}(s) + \eta_{i}$ [cite: 3797]
    * Where $f_i(s)$ is the mean response of neuron 'i' to stimulus 's'.
    * $\eta_i$ is noise, with $\eta \sim \mathcal{N}(0, Q(s))$. $Q(s)$ is the noise covariance matrix. [cite: 3797]

**Likelihood of Observing Response Vector r:**
* $P[r|s] = \frac{1}{\sqrt{(2\pi)^{N}det Q(s)}} \exp\left(-\frac{1}{2}[r-f(s)]^{T}Q^{-1}(s)[r-f(s)]\right)$ [cite: 3797]
    * This is the probability density function for a multivariate Gaussian distribution.

**Fisher Information $I_F(s)$:**
* The Fisher information consists of two terms:
    $I_{F}(s) = f^{\prime}(s)^{T}Q^{-1}(s)f^{\prime}(s) + \frac{1}{2}Tr[Q^{\prime}(s)Q^{-1}(s)Q^{\prime}(s)Q^{-1}(s)]$ (The slide has $Q^{-1}(s)$ appearing three times in the trace term's denominator, which looks like a typo. Standard FI for Gaussian with mean $f(s)$ and covariance $Q(s)$ when Q also depends on s involves derivatives of Q. The formula in the slide seems to be: $I_F(s) = f'(s)^T Q^{-1}(s) f'(s) + \frac{1}{2} Tr[(Q'(s)Q^{-1}(s))^2]$) [cite: 3797]
    * $f^{\prime}(s)$ is the derivative of the mean response with respect to 's' (how tuning curves change with stimulus).
    * $Q^{\prime}(s)$ is the derivative of the noise covariance matrix with respect to 's'.
    * The first term relates to how the mean response changes with the stimulus.
    * The second term relates to how the noise covariance changes with the stimulus (often ignored if Q is constant or its change is small). [cite: 3797]

**Notes:**
This slide introduces Fisher Information as a measure of coding accuracy for Gaussian population codes. It defines the model for individual neuron responses (mean response plus Gaussian noise) and the likelihood of observing a particular population response vector. The formula for Fisher Information is presented, highlighting its two components: one related to changes in the mean response and the other to changes in noise covariance with the stimulus.

## Slide 8



---
### Additive Correlated Noise
---

**Assumption:**
* Additive noise: Q (noise covariance matrix) is independent of s. [cite: 3798]

**Simplified Fisher Information:**
* If Q is independent of s, then $Q'(s) = 0$, and the second term in the Fisher Information formula vanishes.
    $I_{F}(s) = f^{\prime}(s)^{T}Q^{-1}f^{\prime}(s)$ [cite: 3799]

**Scenario:**
* All neurons have variance $\sigma^2$. [cite: 3799]
* Pairwise correlation is 'c'. [cite: 3799]

**Covariance Matrix Q and its Inverse $Q^{-1}$:**
* $Q_{ij} = \sigma^{2}[\delta_{ij} + c(1-\delta_{ij})]$ [cite: 3800]
    * This is the same correlation structure as defined on Slide 5.
* $Q_{ij}^{-1} = \frac{\delta_{ij}(Nc+1-c)-c}{\sigma^{2}(1-c)(Nc+1-c)}$ [cite: 3800]
    * This gives the elements of the inverse of the covariance matrix.

**Definitions for Fisher Information Expansion:**
* $F_{1}(s) = \frac{1}{N}\sum_{i}(f_{i}^{\prime}(s))^{2}$ (Average of squared derivatives of individual tuning curves) [cite: 3800]
* $F_{2}(s) = \left(\frac{1}{N}\sum_{i}f_{i}^{\prime}(s)\right)^{2}$ (Square of the average derivative of tuning curves) [cite: 3800]

**Fisher Information $I_F(s)$ under these conditions:**
* $I_{F}(s) = \frac{N[F_{1}(s)(1-c) + cN(F_{1}(s)-F_{2}(s))]}{\sigma^{2}(1-c)(Nc+1-c)}$ (The slide shows a slightly different numerator structure after expansion: $cN^2[F_1(s)-F_2(s)] + (1-c)NF_1(s)$ which is equivalent to $N F_1(s) (1-c+cN) - cN^2 F_2(s)$). [cite: 3800]
* As $N \rightarrow \infty$, the Fisher information approaches:
    $I_{F}(s) \rightarrow \frac{N[F_{1}(s)-F_{2}(s)]}{\sigma^{2}(1-c)}$ (This seems to be a specific limit, perhaps for large $N$ where $Nc \gg 1-c$. More generally, as $N \rightarrow \infty$, if $c \ne 1$, the term $Nc+1-c \approx Nc$. Then $I_F(s) \approx \frac{N F_1 (1-c) + cN^2(F_1-F_2)}{\sigma^2(1-c)Nc} = \frac{F_1(1-c)/c + N(F_1-F_2)}{\sigma^2(1-c)}$)
    *Self-correction: The slide's given limit is for $Nc \gg 1-c$, so $Q^{-1}$ becomes simpler. Let's re-derive $I_F(s)$ based on the provided expression in the slide for large N. $I_F(s) = \frac{cN^2(F_1-F_2) + (1-c)NF_1}{\sigma^2(1-c)(Nc+1-c)}$. For large N, $(Nc+1-c) \approx Nc$. So $I_F(s) \approx \frac{cN^2(F_1-F_2) + (1-c)NF_1}{\sigma^2(1-c)Nc} = \frac{N(F_1-F_2)}{\sigma^2(1-c)} + \frac{F_1}{\sigma^2 c}$. The limit in the slide is $\frac{N[F_1-F_2]}{\sigma^2(1-c)}$. This means the $NF_1(s)$ term dominated the $cN^2(F_1-F_2)$ for small c, or the $(1-c)$ term in the denominator is critical.*
    The slide's implication $I_F(s) \rightarrow \frac{N[F_1-F_2]}{\sigma^2(1-c)}$ seems to be reached by asserting that as N grows, the $cN^2(F_1-F_2)$ term dominates the $(1-c)NF_1(s)$ term in the numerator, and $Nc+1-c \approx Nc$ in the denominator.
    So $I_F(s) \approx \frac{cN^2(F_1-F_2)}{\sigma^2(1-c)Nc} = \frac{N(F_1-F_2)}{\sigma^2(1-c)}$. This limit means that information increases linearly with N provided $F_1 > F_2$. [cite: 3800]

**Interpretation:**
* The more heterogeneous the derivatives $f_{i}^{\prime}(s)$ are (i.e., $F_{1} > F_{2}$), the more information is preserved. [cite: 3800]
* Fisher information is also growing with N (number of neurons) and with c (correlation coefficient, assuming $F_1 > F_2$). [cite: 3801]

**Notes:**
This slide simplifies the Fisher Information calculation by assuming additive noise (Q is independent of s). It then explores a specific scenario where neurons have uniform variance and pairwise correlation 'c'. The resulting Fisher Information formula shows that information increases with N and benefits from heterogeneity in tuning curve derivatives ($F_1 > F_2$). The effect of correlation 'c' is also highlighted, showing it can increase information under these conditions.

## Slide 9



---
### Limited-Range Correlations
---

**More Realistic Case:**
* Consider limited-range correlations that decay with the distance between neurons. [cite: 3802]

**Covariance Matrix Structure $Q_{ij}$:**
* $Q_{ij} = \sigma^{2}\rho^{|i-j|}$ [cite: 3803]
    * Where $\sigma^2$ is the variance of individual neurons.
    * $\rho = \exp(-\Delta/L)$ is the correlation coefficient between adjacent neurons. [cite: 3803]
        * $\Delta$ is the spacing between peaks of adjacent tuning curves. [cite: 3803]
        * L is a characteristic correlation length. [cite: 3803]
    * $|i-j|$ is the distance between neuron 'i' and neuron 'j'.

**Fisher Information in the Limit $N \rightarrow \infty$:**
* $I_{F}(s) \rightarrow \frac{N(1-\rho)F_{1}(s)}{\sigma^{2}(1+\rho)}$ [cite: 3803]
    * Where $F_1(s)$ is the average of squared derivatives (as defined on Slide 8).

**Interpretation:**
* $I_F$ still grows linearly with N. [cite: 3803]
* Higher correlation ($\rho \rightarrow 1$) reduces $I_F$: shared noise limits how much signal averaging helps. [cite: 3803]
* Correlations decay with distance, so only nearby neurons interfere. [cite: 3803]

**Summary for Limited-Range Correlations:**
* Correlations reduce the efficiency of information scaling (the factor multiplying N is smaller than if $\rho=0$). [cite: 3803]
* However, they do not prevent the Fisher information from growing linearly with the number of encoding neurons. [cite: 3803]

**Notes:**
This slide introduces a more realistic model of neural correlation where the correlation strength decays with the distance between neurons. The Fisher Information for this "limited-range" correlation model still grows linearly with N, but the rate of growth is reduced by the correlation parameter $\rho$. Stronger correlations (larger $\rho$) lead to a smaller increase in information with N.

## Slide 10



---
### Noise and Signal Covariance
---

**Law of Total Covariance:**
* $cov[r] = cov[E[r|s]] + E[cov[r|s]]$ (The slide shows $cov[r]=cov[E[r|s]+E[cov[r|s]]$, which seems to be a typo. The law of total covariance is $Var(Y) = E[Var(Y|X)] + Var(E[Y|X])$. In terms of covariance matrices for a response vector $r$ and stimulus $s$, this is $Cov(r) = E_s[Cov(r|s)] + Cov_s(E[r|s])$.) [cite: 3804]
    * $Cov_s(E[r|s])$ is called **signal covariance**. [cite: 3804]
    * $E_s[Cov(r|s)]$ is called **noise covariance**. [cite: 3804]

**Application to Neuron Responses:**
* In our typical setup, $E[r|s] = f(s)$ (the mean response is the tuning curve vector). [cite: 3805]
* So we have:
    * $cov_{signal} = E_{s}[f(s)f(s)^{T}] - E[f(s)]E[f(s)]^{T}$ [cite: 3805]
        * This measures how the mean responses $f(s)$ co-vary as the stimulus 's' changes. $E_s$ denotes expectation over stimuli.
    * $cov_{noise} = E_{s}[\Sigma(s)]$ [cite: 3805]
        * This is the average noise covariance, averaged over stimuli. $\Sigma(s)$ here is $Cov(r|s)$.

**Notes:**
This slide decomposes the total covariance of neural responses into two components using the law of total covariance: signal covariance and noise covariance. Signal covariance arises from the variability in mean responses due to changes in the stimulus, while noise covariance arises from the trial-to-trial variability of responses to a fixed stimulus, averaged across stimuli.

## Slide 11



---
### Noise Correlations in More Detail
---

**(Image: Three plots based on Cohen and Kohn, "Measuring and interpreting neuronal correlations" [cite: 3806])**

* **(Plot a) Tuning Curves:** [cite: 3807]
    * Shows tuning curves for two neurons (Cell 1 and Cell 2) for "Direction" (stimulus).
    * Open circles represent mean responses.
    * Small points show responses to individual presentations of a stimulus at a particular direction.
    * Illustrates variability in responses even for the same stimulus.

* **(Plot b) Spike Count Correlation (Noise Correlation):** [cite: 3808]
    * Scatter plot of "Response cell 2" vs "Response cell 1" (spikes per s) for *the same stimulus*.
    * $r_{SC} = 0.21$ indicates a positive noise correlation.
    * Measures the correlation between fluctuations in responses of the two cells to the *same* stimulus.

* **(Plot c) Signal Correlation:** [cite: 3809]
    * Scatter plot of "Response cell 2" vs "Response cell 1" (spikes per s) for *different stimuli*.
    * $r_{signal} = -0.11$ indicates a negative signal correlation.
    * Measures the correlation between the two cells' *mean responses* to different stimuli. [cite: 3809]
    * Each point represents the mean response to a given direction of motion. [cite: 3810]
    * Signal correlation is negative because Cell 2's responses increase over a range of motion directions where Cell 1's responses decline. [cite: 3811]

**Summary Points:**
* **(a) Tuning curves** illustrate how individual neurons respond on average to different stimuli, and also show trial-to-trial variability for repeated presentations of the same stimulus. [cite: 3807]
* **(b) Noise correlations** quantify how the trial-to-trial fluctuations of different neurons are related when responding to the *same* stimulus. A positive $r_{SC}$ means that when one neuron fires more than its average for a given stimulus, the other tends to do so as well. [cite: 3808]
* **(c) Signal correlations** quantify how the *average responses* of different neurons to *different* stimuli are related. A negative $r_{signal}$ means that stimuli that elicit a strong response from one neuron tend to elicit a weak response from the other. [cite: 3809, 3811]

**Notes:**
This slide provides a visual and conceptual explanation of noise correlations and signal correlations using data from Cohen and Kohn (2008). It distinguishes between how neurons' variable responses to the *same* stimulus co-vary (noise correlation) and how their average responses across *different* stimuli co-vary (signal correlation).

## Slide 12



---
### Noise Correlations Again, From a More Theoretical View
---

**(Image: Based on Averbeck, Latham, and Pouget, "Neural correlations, population coding and computation" [cite: 3816])**

* **(a) Tuning Curves:** [cite: 3812]
    * Schematic tuning curves for Neuron 1 and Neuron 2.
    * Mean response (spikes) vs. Stimulus.
    * Indicates $S_1$ as a specific stimulus.

* **(b) Examples of Noise Correlation at $s_1$:** [cite: 3812]
    * **Left (Positive Noise Correlation):** Scatter plot for Neuron 1 and Neuron 2 responses to stimulus $s_1$. Responses tend to fluctuate together (if N1 is above its mean, N2 tends to be too).
    * **Right (Negative Noise Correlation):** Scatter plot for Neuron 1 and Neuron 2 responses to stimulus $s_1$. Responses tend to fluctuate oppositely (if N1 is above its mean, N2 tends to be below).

* **(c) Uncorrelated Population Response:** [cite: 3812, 3814]
    * Plot of population response where the x-axis is the preferred orientation of neurons and y-axis is their response. [cite: 3813]
    * Neurons exhibit noise fluctuations.
    * On individual trials, the responses of nearby neurons are uncorrelated (fluctuating up and down independently). [cite: 3814]

* **(d) Correlated Population Response:** [cite: 3812, 3814, 3815]
    * Similar plot to (c).
    * On individual trials, the responses of nearby neurons are correlated (fluctuating up and down together). [cite: 3814]
    * Nearby neurons are positively correlated (like panel b, left). [cite: 3815]
    * Neurons that are far apart (in preferred orientation) are negatively correlated (like panel b, right). [cite: 3815]

**Source:** Averbeck, Latham, and Pouget, "Neural correlations, population coding and computation". [cite: 3816]

**Notes:**
This slide offers a more theoretical illustration of noise correlations, building on the concepts from the previous slide and referencing Averbeck et al. (2006). It shows how positive and negative noise correlations for a pair of neurons might look, and then extends this to a population level, contrasting uncorrelated noise with spatially structured correlated noise (positive for nearby neurons, negative for distant ones).

## Slide 13: (Intuition for the effect of correlation on encoding accuracy - Part 1)



---
### Intuition for the Effect of Correlation on Encoding Accuracy
---
**(Image: Complex figure with multiple scatter plots, illustrating decision boundaries. Based on Averbeck, Latham, and Pouget, "Neural correlations, population coding and computation" [cite: 3773])**

**Scenario:**
* Consider two neurons and two stimuli ($S_1, S_2$) that we want to encode such that they can be discriminated later. [cite: 3771]
* The plots show responses of Neuron 2 vs. Neuron 1 (spikes). Ellipses represent the distribution of responses to $S_1$ and $S_2$.
* $W_{diag}$ represents a decision boundary (discriminator) based on shuffled responses (ignoring correlations).
* $W_{optimal}$ represents a decision boundary based on unshuffled responses (accounting for correlations).

**Method:**
* To understand the effect of correlation, we create shuffled data where response correlations are removed. [cite: 3772]
* We estimate $W_{diag}$ (a simple diagonal discriminator) on these shuffled responses.
* Then, we apply this $W_{diag}$ to the original, unshuffled (correlated) responses to measure its performance ($I_{diag}$).
* $\Delta I_{shuffled} = I - I_{shuffled}$, where $I$ represents some measure of information or discriminability. The line shown is the best discrimination threshold for an ideal observer who knows about correlations. [cite: 3773]

**Panel (a): $\Delta I_{diag} = 0$** [cite: 3770]
* Shuffled responses show some separation. $W_{diag}$ is estimated.
* Applying $W_{diag}$ to unshuffled responses shows $W_{diag} = W_{optimal}$.
* In this case, correlations do not hurt (or help significantly) the performance of a simple diagonal decoder. The information lost by ignoring correlations is zero.

**Panel (b): $\Delta I_{shuffled} > 0$** [cite: 3770, 3771]
* Shuffled responses show clear separation. $W_{diag}$ is estimated.
* Applying $W_{diag}$ to unshuffled responses shows it is *not* optimal. $W_{optimal}$ would be different.
* The "information" measure from the shuffled data is greater than from the correlated data when using $W_{diag}$. Correlations here are detrimental if ignored by the decoder. An optimal decoder, however, might still perform well.

**Panel (c): $\Delta I_{shuffled} = 0$** [cite: 3770, 3771]
* Shuffled responses show some separation. $W_{diag}$ is estimated.
* Applying $W_{diag}$ to unshuffled responses shows $W_{diag} = W_{optimal}$.
* Similar to (a), correlations do not significantly impact the performance of this specific decoder.

**Source:** Averbeck, Latham, and Pouget, "Neural correlations, population coding and computation". [cite: 3773]

**Notes:** This slide begins to build intuition about how correlations affect encoding accuracy by considering how a simple decoder (trained on data where correlations are removed) performs on correlated data. It introduces scenarios where ignoring correlations might not matter (a) or might be detrimental (b) for such a decoder. The ideal observer knows about correlations.

## Slide 14: (Intuition for the naive decoding - Part 2)



---
### Intuition for the Naive Decoding
---
**(Image: Continuation of the scatter plots from Slide 13, focusing on a "naive" decoder that doesn't know about correlations.)**

**Scenario (Continued):**
* Again, two neurons and two stimuli. Neuronal responses are correlated.
* Imagine that the decoder *does not know* about the correlations and creates decision lines ($W_{diag}$) based on shuffled (correlation-free) responses. [cite: 3775]

**Panel (a) from previous slide, relabeled for naive decoding: $\Delta I_{diag} = 0$** [cite: 3774]
* Naive decoder estimates $W_{diag}$ on shuffled data.
* Applies $W_{diag}$ to unshuffled (correlated) data.
* In this specific scenario, $W_{diag}$ happens to be $W_{optimal}$.
* The naive decoder performs optimally despite ignoring correlations. Performance is kept. [cite: 3775]

**Panel (b) from previous slide, relabeled for naive decoding: $\Delta I_{diag} > 0$** [cite: 3774]
* Naive decoder estimates $W_{diag}$ on shuffled data.
* Applies $W_{diag}$ to unshuffled (correlated) data.
* Here, $W_{diag} \ne W_{optimal}$.
* The naive decoder performs sub-optimally. Performance is destroyed by ignoring the correlations. [cite: 3775]

**Summary from figures:**
* A naive decoder (one that ignores correlations by training on shuffled data) can sometimes maintain its performance on correlated data (as in panel a), or its performance can be destroyed (as in panel b) if the correlation structure makes the stimuli less separable along the dimension chosen by the naive decoder. [cite: 3775]

**Overall Summary (from text at bottom of slide image):**
* Correlations are inevitable in the responses of neurons. [cite: 3775]
* In repetitive trials, we can split noise and signal correlations. [cite: 3775]
* Noise correlations can have a detrimental but sometimes also helpful effect. [cite: 3775]
* **Next time:** Examples of noise correlations studies, Neural manifolds. [cite: 3775]

**Notes:** This slide continues the intuition-building by focusing on a "naive" decoder that is unaware of the correlation structure. It shows that such a decoder's performance can either be unaffected or severely degraded by the presence of correlations, depending on how the correlations interact with the stimulus distributions. The slide concludes with a summary of the inevitability of correlations and their potential effects.

## Slide 15: (No content, likely just "Summary" as a title)

---
### Summary (Implicit based on previous slide's text)
---
* Correlations are an inherent feature of neural responses.
* It's possible to distinguish between noise correlations (variability to the same stimulus) and signal correlations (variability in mean response across different stimuli).
* Noise correlations can impair or, in some situations, enhance the accuracy of population codes.

**Further Topics:**
* Examples of studies on noise correlations.
* Neural manifolds.

**Notes:** This slide would typically summarize the key takeaways from the lecture on correlated variability.

## Slide 16: Bibliography



---
### Bibliography
---

* Abbott, L. F. and Peter Dayan. "The effect of correlated variability on the accuracy of a population code". In: Neural Computation 11.1 (1999), pp. 91-101. [cite: 3776]
* Averbeck, Bruno B, Peter E Latham, and Alexandre Pouget. "Neural corre- lations, population coding and computation". In: Nature reviews neuroscience 7.5 (2006), pp. 358-366. [cite: 3777, 3778]
* Cohen, Marlene R. and Adam Kohn. "Measuring and interpreting neuronal correlations". In: Nature Neuroscience 14.7 (2011), pp. 811-819. arXiv: NIHMS150003. [cite: 3779]
* Shadlen, Michael N and William T Newsome. "Noise, neural codes and cortical organization". In: Current Opinion in Neurobiology 4.4 (Aug. 1994), pp. 569-579. ISSN: 09594388. [cite: 3780, 3781]

**Notes:** This slide lists key references related to the topics discussed in the lecture on neural correlations and population coding.