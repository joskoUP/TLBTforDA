# Time-limited Balanced Truncation for unstable Bayesian Inference

This MATLAB repository contains code for the numerical results of the following paper:

1. König, J., Freitag, M. "Time-limited Balanced Truncation for Data Assimilation Problems" [arXiv](https://arxiv.org/abs/2212.07719) (submitted, under review).

## Summary
The work in [1] generalizes the concept of balancing for Bayesian inference from [2] to arbitrary prior covariances and unstable system matrices. For this purpose, the concept of time-limited balanced truncation is used. Time-limited Gramians are efficiently computed by rational Krylov methods from [4]. Numerical examples compare the performance with the original approach from [2] and the optimal approach from [3]. 
This work uses the code from [2] 
(to be found at [https://github.com/elizqian/balancing-bayesian-inference](https://github.com/elizqian/balancing-bayesian-inference)) and [4] (to be found at [https://zenodo.org/record/7366026](https://zenodo.org/record/7366026)).

## Examples
To run this code, you need the MATLAB Control System Toolbox.

To generate the plots comparing computations with a compatible and non-compatible prior from the paper (Figure 1), run the heat_incompatible_prior.m script.

To generate the TLBT plots from the paper, run the compare_*.m scripts, corresponding to:
* compare_heat.m: The heat equation example for end times T = 1, 3, 10 with measurements spaced 0.005 seconds apart (Figure 2)
* compare_ISS.m: The ISS example for end times T = 1, 3, 10 with measurements spaced 1 seconds apart (Figure 3)
* compare_unstable.m: The advection-diffusion equation example for end times T = 0.1, 0.5, 1 with measurements spaced 0.001 seconds apart (Figure 4)

## References
2. Qian, E., Tabeart, J. M., Beattie, C., Gugercin, S., Jiang, J., Kramer, P. R., and Narayan, A.
"[Model reduction for linear dynamical systems via balancing for Bayesian inference](https://link.springer.com/article/10.1007/s10915-022-01798-8)." Journal of Scientific Computing 91.29 (2022).
3. Spantini, A., Solonen, A., Cui, T., Martin, J., Tenorio, L., and Marzouk, Y. "[Optimal low-rank approximations of Bayesian linear inverse problems](https://epubs.siam.org/doi/pdf/10.1137/140977308?casa_token=CaYk5XimLkoAAAAA:-WjPu7U7kT8q3WZU66efl5X6GPylJOcnJM7XuOyy-I00LLa0vo9478Tv4BeNFoO67EwOsvl78Q)." SIAM Journal on Scientific Computing 37. 6 (2015): A2451-A2487.
4. Kürschner, P. "[Balanced truncation model order reduction in limited time intervals for large systems](https://link.springer.com/article/10.1007/s10444-018-9608-6)." Advances in Computational Mathematics 44.6 (2018): 1821–1844.

### Contact
Please feel free to contact [Josie König](https://www.math.uni-potsdam.de/professuren/datenassimilation/personen/josie-koenig) with any questions about this repository or the associated paper.
