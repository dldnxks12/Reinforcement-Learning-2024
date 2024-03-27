### chapter 2 : MAB

---

- k-armed bandit 

        # action-value estimation: simple average ( Q(a) will converge to true Q(a) as n ➔ ∞ )
        ➔ 'Sample Average' Method
  
        # action-selection: greedy ➔ exploitation only... we need 'exploration'...!  
        ➔ ε-greedy (exploration + exploitation)
            ➔ all the Q values will converge to true Q as n ➔ ∞

- Incremental update

        # Sample Average를 사용할 때, 모든 데이터들을 메모리에 저장해두었다가 계산하는 건 too expensive (memory/computation)
        ➔ use 'incremental update' 
            ➔ Q(n+1) = Q(n) + (1/n)[R(n) - Q(n)]
            ➔ New-estimation = Old-estimation + Step-size[Target - Old-estimation]
    

- Non-stationary problem 

        # Non-stationary problem ➔ dynamics change over time
                *In Bandit : reward probabilities
                *In MDP    : reward probabilities and transition kernel.
  
          ➔ so.. let's give weight to the recent reward!
              ➔ Use a constant step size (α) ➔ i.e. Q(n+1) = Q(n) + (α)[R(n) - Q(n)], where α is constant.
              (It naturally replaces the 'Sample Average' ➔ 'Weighted Average')
    
        # Conditions of step size, required to converge ➔ Robbins-Monro Rule

          ➔ Sample Average (1/n): meets Robbins-Monro ➔ converge (Law of Large Number)
                  ➔ But it is slow and difficult to fine-tune
                          ➔ Used in theoretical works.
                          ➔ Not useful in practice (application / empirical research).
          ➔ Weighted Average (α): violates Robbins-Monro ➔ not converge
                  ➔ But useful for non-stationary problems.


- Initial bias


        Sample Average
                  ➔ No initial bias
                  ➔ But, not practical in non-stationary problems
  
        Weighted Average
                  ➔ Induce initial bias
                  ➔ But, practical in non-stationary problems
    
        * Address the initial bias and non-stationary problems?
        ➔ Use Unbiased Constant Step-size Trick!


- Upper Confidence Bound (UCB)

        # Action-selection method.
        ➔ Greedy   : No exploration
  
        ➔ ε-greedy : Do exploration among the non-greedy actions, but no preference.
                  (i.e. randomly select actions among non-greedies)
  
        ➔ UCB      : Select non-greedy actions according to their potential. 
                
                ➔ UCB fails to work in non-stationary problems!
                ➔ UCB fails to work in large-state space!
                ➔ Bandit problem이 아닌 일반적인 RL 문제에 사용하기엔 ε-greedy가 더 낫다.
    
                * Improved version of UCB
                  ➔ UCB 1  (if we dont have a prior knowledge about reward distribution)
                  ➔ KL-UCB (if we have a prior knowledge about reward distribution)
            
    
---

    
    


    

    







    


    
