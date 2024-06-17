#### Temporal difference (TD)

---

- MC vs TD


      MC and TD 둘 다 stochastic approximation of Bellman Eqns.  
      * Bellman Eqn. : Q(s, a) = E[Gt | St=s, At=a] = E[r + γQ(s', a') | St=s, At=a]

      # Monte Carlo

          Q(s, a) = E[Gt | St=s, At=a] 를 stochastic하게 근사
  
          Q(s, a) ← Q(s, a) + α[Gt - Q(s, a)]
      
          ➔ Error is not propagated (no bootstrapping)
          ➔ Bias가 없어서, optimal policy를 찾을 수 있음 (off-policy MC)
          ➔ Variance가 커서 수렴이 늦음.
            
      # Temporal Difference 

          1) SARSA (BEE 근사)
          Q(s, a) = E[r + γQ(s', a') | St=s, At=a] 를 stochastic하게 근사
  
          Q(s, a) ← Q(s, a) + α[r+γQ(s', a') - Q(s, a)]

          2) Q learning (BOE 근사)
          Q(s, a) = E[r + γmaxQ(s', A) | St=s, At=a] 를 stochastic하게 근사
  
          Q(s, a) ← Q(s, a) + α[r+γmaxQ(s', A) - Q(s, a)]
          
          ➔ Error is propagated (bootstrapping)
          ➔ Bias가 있어서 optimal policy를 보장 못함.  
          ➔ 대신, variance가 작아서 빠르게 어떤 값으로 수렴은 함.


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
