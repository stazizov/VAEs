### readme in progress

# Variational Inference

By using variational inference we are trying to approximate hard to compute:

$$
p(z|x) = \frac{p(z)p(x|z)}{\int{p(z)p(x|z)dz}}
$$

So $\int{p(z)p(x|z)dz} = p(x)$ is intractable and we are trying to approximate it by using $q(z)$

by trying to optimize it’s parameters $\theta$. 

The objective for $\theta$ optimization is KL divergence, that allows us to disregard denominator:

$$
KL(q(z)||p(z|x)) = \int{q(z)log(\frac{p(z|x)}{q(z)})}
$$

We are only optimizing **ELBO (Evidence Lower BOund)** that depends on $\theta$.

```math
KL(q(z)||p(z|x)) = \int{q(z)log(\frac{p(z|x)}{q(z)})} \newline \int{q(z)log(p(z|x) - \int{q(z)log(q(z))}}  = \newline \mathbb{E}_{q\approx z} [log(q(z)] -\mathbb{E}_{q\approx z} [log(p(z|x)]
```


note that here we the formula of **Math Expectation of Random Variable**

$$
\mathbb{E}[f(z)] = \int_{-\infty}^{\infty} f(z) \cdot q(z) \, dz
$$

Where f(z) is the function we want to weight in this case it can represent both $log(p(z|x)$ and  $log(q(z)$, and $q(z)$ is the probability density of the random variable $z$.

Finally we have this: 

```math
\mathbb{E}_{q\approx z} [log(q(z)] - \mathbb{E}_{q\approx z} [log(p(z|x)]
```

[Bayes’ theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem):

$$
p(a|b) = \frac{p(b|a)p(a)}{p(b)}
$$

And we can apply [Bayes’ theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) to second term and transform it into: 

 

```math
\mathbb{E}_{q\approx z} [log(q(z)] - \mathbb{E}_{q\approx z} [log(\frac{p(x|z)p(z)}{p(x)})] = \newline = \mathbb{E}_{q\approx z} [log(q(z)] - \mathbb{E}_{q\approx z}[log(p(x|z)p(z))] + log(p(x))
```

So, it’s important to note that $log(p(x))$  doesn’t change, so we can just ignore it.

Finally:

```math
KL(q(z)||p(z|x)) = \mathbb{E}_{q\approx z} [log(q(z)] - \mathbb{E}_{q\approx z} [log(p(x|z)p(z))]
```

Minimising KL divergence is equal to maximising the ELBO

```math
ELBO = -KL = \mathbb{E}_{z \approx q_{\theta}(z)} [log(p(x|z)p(z))- log(q_{\theta}(z))]
```

We can apply gradient or coordinate VI to maximise ELBO

# Reparameterization trick

But Authors wanted more general purpose algorithm based on SGB, for this reason we need to compute gradient of ELBO: 

```math
\nabla_{\theta}\mathbb{E}_{z \approx q_{\theta}(z)} [log(p(x|z)p(z))- log(q_{\theta}(z))] = \newline = \nabla_{\theta} \int q_{\theta}(z)[log(p(x|z)p(z))- log(q_{\theta}(z)]dz 
```

how can we push gradient into integral? 
Let’s apply Leibniz’s Rule: 

$$
\int \nabla_{\theta}[q_{\theta}(z)[log(p(x|z)p(z))- log(q_{\theta}(z))]dz]  = \newline \int \nabla_{\theta}q_{\theta}(z)[log(p(x|z)p(z))- log(q_{\theta}(z))]dz + \newline +  \int \nabla_{\theta}[log(p(x|z)p(z))- log(q_{\theta}(z))]q_{\theta}(z)dz
$$

Let’s look at the first part of second integral, it’s independent and can be removed

$$
\int \nabla_{\theta}log(q_{\theta}(z))q_{\theta}(z)dz
$$

next:

$$
\int \nabla_{\theta}q_{\theta}(z)[log(p(x|z)p(z))- log(q_{\theta}(z))]dz + \newline +  \int \nabla_{\theta}log(q_{\theta}(z))q_{\theta}(z)dz 
$$

let’s simplify: 

$$
\nabla_{\theta}log(q_{\theta}) = \frac{\nabla_{\theta}q_{\theta}(z)}{q_{\theta}(z)}
$$

so then: 

$$
\int \frac{\nabla_{\theta}q_{\theta}(z)}{q_{\theta}(z)}q_{\theta}(z)dz  = \int {\nabla_{\theta}q_{\theta}(z)dz} \newline \nabla_{\theta} \int q_{\theta}(z)dz = \nabla_{\theta}  * 0 = 0
$$

note that: 

$$
\nabla_{\theta}q_{\theta}(z) = \nabla_{\theta}log(q_{\theta})q_{\theta}(z) 
$$

after all the simplification:

```math
\int \nabla_{\theta}q_{\theta}(z)[log(p(x|z)p(z))- log(q_{\theta}(z))]dz + \newline +  \int \nabla_{\theta}log(q_{\theta}(z))q_{\theta}(z)dz =  \newline
\int \nabla_{\theta}q_{\theta}(z)[log(p(x|z)p(z))- log(q_{\theta}(z))]dz =  \newline
\int \nabla_{\theta}log(q_{\theta})q_{\theta}(z) [log(p(x|z)p(z))- log(q_{\theta}(z))]dz \approx \newline \approx \nabla_{\theta}\mathbb{E}_{z \approx q_{\theta}(z)} [log(p(x|z)p(z))- log(q_{\theta}(z))] = \newline \frac{1}{L}\sum_{l=1}^{L}{log(p(x|z_l)p(z_l))- log(q_{\theta}(z_l))}
```

But the problem is that this gradient have high variance and are useless.

So the authors invented reparameterization trick:

So the deconstructed $q_{\theta}(z)$  to some standart distribution that does not depends on the parameters $\theta$  and differentiable transformation $\epsilon$

- $\epsilon \approx p'(\epsilon)$
- $z = g_{\theta}(\epsilon, x)$

```math
\nabla_{\theta}\mathbb{E}_{z \approx q_{\theta}(z)} [log(p(x|z)p(z))- log(q_{\theta}(z))] = \newline = \nabla_{\theta}\mathbb{E}_{p'} [log(p(x|g_{\theta}(\epsilon, x))p(g_{\theta}(\epsilon, x)))- log(q_{\theta}(g_{\theta}(\epsilon, x)))] \newline \approx \nabla_{\theta}\sum_{l}^{L} [log(p(x|g_{\theta}(\epsilon_l, x))p(g_{\theta}(\epsilon_l, x)))- log(q_{\theta}(g_{\theta}(\epsilon_l, x)))]
```

When can we use reparameterization trick?

- Location scale of distributions
    
    we have a Normal Distributions, lets write $z$ as: 
    
    
    $X(z) = \sigma^2 * z + \mu \approx \N(\mu, \sigma^2)$
    
    

- When we have tractable CDF that we can invert
    
    for example we have exponential distribution
    
    - $x = Exp(x)$
    - $u = U(0, 1)$
    - define transformation with parameter $\lambda$
    - $X = -\frac{log(1 - u)}{\lambda}$
    - $F_X(x) = P(X \leq x) = P(-\frac{log(1 - u)}{\lambda} \leq x) = P(u \leq 1 - e^{-\lambda x}) =  1 - e^{-\lambda x}$
    
    So, this is a CDF of exponential distribution
    
    - $X = -\frac{ln(1 - u)}{\lambda}$
- Composition
    - E.g we have a Gamma
    - $u = U(0, 1)$
    - $X = -\frac{log(1 - u)}{\lambda}$
    - $S(x_1. ..x_n) = \sum_{i=1}^{n}{s_i} \approx Gamma(n, \lambda)$
