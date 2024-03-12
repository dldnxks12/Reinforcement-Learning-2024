### chapter 2 : MAB

---

- k-armed bandit 

        # action-value estimation: simple average (Q(a) will converge to true Q(a) as n ➔ ∞)
        ➔ 'Sample Average' Method
  
        # action-selection: greedy ➔ exploitation only... we need 'exploration'...!  
        ➔ ε-greedy (exploration + exploitation)
            ➔ all the Q values will converge to true Q as n ➔ ∞

- Incremental update

        # simple average를 사용할 때, 모든 데이터들을 메모리에 저장해두었다가 계산하는 건 too expensive (memory/computation)
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


        # From here : 0313
  
        # Initial bias
            ➔ Sample average     ➔ No initial bias
            ➔ Constant step size ➔ Induce initial bias
    
                To address the initial bias problem in non-stationary problems
                ➔ Use Unbiased Constant Step-size Trick


- Upper Confidence Bound (UCB)

    
        # Why do we need to explore?
        ➔ Exploration is crucial ➔ uncertainties about the accuracy of the action-value estimations.
    
        # Action-selection method.
        ➔ Greedy   : No exploration
        ➔ ε-greedy : Do exploration among the non-greedy actions, but without preference. i.e. randomly.
        ➔ UCB      : Select non-greedy actions according to their potential. 
    
                ➔ UCB is good, but it fails to work in non-stationary problems
                    ➔ bandit problem이 아닌 일반적인 RL 문제에 사용하기엔 ε-greedy가 더 낫다.
    
---

    
    


    

    







    


    
