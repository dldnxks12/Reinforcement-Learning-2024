#### Temporal difference (TD)

---

- MC vs TD


    MC and TD are all stochastic approximation of true value
    i.e., V(s) = E[Gt | St=s] = E[r + γV(s') | St=s] 

    * Monte Carlo: V(s) ⭠ V(s) + α[Gt - V(s)]

        Update target : Gt
        ➔ Error is not propagated (Each value estimation is independent)
        ➔ MC can be guaranteed to be converged to optimal value

    * Temporal Difference: V(s) ⭠ V(s) + α[r+γV(s') - V(s)]

        Update target : r + γV(s')
        ➔ Error is propagated

    and... TD also can be guaranteed to be converged to optimal value?????


    *************************************************************************

    MC는 TD와 달리 bias가 없어서 optimal optimal policy를 찾을 수 있다.
    TD는 MC에 비해 variance가 낮아서 빠르게 수렴하지만, bias가 있어 optimal을 보장하지 못한다.

    *************************************************************************


--- 

- On-policy TD control : SARSA



    SARSA : Stochastic approximation of Bellman Equation.

    Q(s, a) = E[r + γQ(s', a') | St=s, At=a]  : Bellman Eqn.   
    
    Q(s, a) ⭠ Q(s, a) + α[r + γQ(s', a') - Q(s, a)] : SARSA 

    ➔ We use Q instead of V, because we dont know the dynamics!


    * Expected SARSA

    Q(s, a) ⭠ Q(s, a) + α[r + γE[Q(s', A)|s'] - Q(s, a)]
    
    ➔ It increases the computational burden, but decrease the variance!


- Convergence of SARSA


    SARSA converges to the 'optimal' q, under the following conditions
      ➔ 1. π satisfies 'GLIE'  
      ➔ 2. step-size α satisfies 'stochastic convergence equations'
        ➔ i..e, α satisfies Robbins-Monro.


---

- Off-policy TD control : Q-Learning


    Q-Learning : Stochastic approximation of Bellman Optimality Equation.

    Q(s, a) = E[r + γmax_{a}_Q(s', a') | St=s, At=a]  : Bellman Opt. Eqn.   
    
    Q(s, a) ⭠ Q(s, a) + α[r + γmax_{a}_Q(s', a') - Q(s, a)] : Q-Learning


- Convergence of Q-Learning

  
    Q-Learning's correct convergence is guranteed.
      ➔ 1. as long as all pairs continue to be updated
      ➔ 2. step-size α satisfies 'stochastic convergence equations'
        ➔ i..e, α satisfies Robbins-Monro.


---


- n-step TD 



    n-step TD = MC + TD