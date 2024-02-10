### chapter 2 : MAB

---

- k-armed bandit 


        # action-value estimation : simple average (converge to true action-value as n ➔ ∞)
        # action-selection : greedy ➔ exploitation only.. we need exploration..!  
            ➔ so, ε-greedy (exploration + exploitation)
            >> all the Q values converge to true action value as n ➔ ∞

- Incremental update
    

        # simple average를 사용할 때, 모든 데이터들을 모아서 한 번에 계산하는 건 expensive in memory/computation.
        ➔ incremental update 
            >> Q(n+1) = Q(n) + (1/n)[R(n) - Q(n)]
            >> New-estimation = Old-estimation + Step-size[Target - Old-estimation]
    

- Non-stationary problem

    
        # Non-stationary problem : reward probabilities changes over time. 
        ➔ give weight to recent reward !
            >> we can use a constant step size (α). i.e. Q(n+1) = Q(n) + (α)[R(n) - Q(n)]
            >> sample average X
    
    
        # Conditions of step size required to converge.
        ➔ Robbins-Monro Rule 
            >> Simple average (1/n) meets the conditions ➔ converge
                ➔ But it it slow and difficult to fine tune ➔ not useful in practical.        
            >> Constant step size (α) violates Robbins-Monro ➔ not converge
                ➔ But useful for non-stationary problems.
    
            
        # Initial bias
            >> Sample average     ➔ No initial bias
            >> Constant step size ➔ Induce initial bias
    
                To address the initial bias problem in non-stationary problems
                ➔ Use Unbiased Constant Step-size Trick

- Upper Confidence Bound (UCB)

    
        # Why do we need to explore?
        ➔ Exploration is crucial ➔ uncertainties about the accuracy of the action-value estimations.
    
        # Action-selection method.
        ➔ Greedy   : No exploration
        ➔ ε-greedy : Do exploration among the non-greedy actions, but without preference. i.e. randomly.
        ➔ UCB      : Select non-greedy actions accroding to their potential. 
    
        >> UCB is good, but it fails to work in non-stationay problems
            >> bandit problem이 아닌 일반적인 RL 문제에 사용하기엔 ε-greedy가 더 낫다.
    
    
---

    
    


    

    







    


    
