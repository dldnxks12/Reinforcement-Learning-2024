### chapter 5 : Monte Carlo Methods

---

- Quick Summary of MC 

      * Assumptions in DP and MC

           - DP : Perfect Model; p(s', r | s, a) is known
           - MC : Perfect Model; p(s', r | s, a) is unknown

      * MC Methods is also GPI framework
  
           - Prediction (policy evaluation)
                 - First visit
                 - Every visit
       
           - Control (policy improvement)
                 - greedy + ES (일반적으로 사용어려움)
                 - ϵ-greedy

      * Near-optimal? use off-policy !!
    
           On-policy MC with ϵ-greedy  ➔ get near-optimal policy
           Off-policy MC with ϵ-greedy ➔ get optimal policy
  
           ➔ DP 처럼 MC 도 optiaml policy를 보장한다!

      * Good and Bad in MC

           - Good : more robust to violation of Markovian property
             (bootstrapping 안해서 그렇다.)
  
           - Bad : only usable in episodic tasks.
  
--- 

- Monte Carlo Prediction (first-visit MC / every-visit MC)

      # Policy evaluation step #

      * DP vs MC
  
        Both of them follow the GPI framework (PE ➔ PI ➔ ...)
  
        However,
  
        DP evaluates v or q by 'computing' from known dynamics of MDP.
        MC evaluates v or q by 'learning' from sample returns in sample episodes.


      * Advantages of MC over DP.
  
        The estimation for each state is independent of other states (unlike DP).

        ➔ The estimation for one state does not build upon the estimation of any other state.
          ➔ No bootstrapping
        ➔ The computational complexity of estimating v, q is independent to the number of states!
          ➔ state 수가 매우 많아져도 괜찮다는 뜻.


      * 그럼 뭘 추정할까.. v? q?

        Under the GPI framework, we need to improve the policy over approximated v or q.
        이를 위해서는 'q'를 추정하는게 더 낫다. 
        
        ➔ 왜? v를 추정하게 되면 policy improvement할 때, transition probability 정보가 필요.. 
            
        근데, q를 추정하면 문제가 하나 발생한다.
        ➔ 모든 state-action pair들이 다 충분히 많이 sampling 되어야한다는 것.
  
        SO?
        1) Use greedy policy + exploring starts (ES)
        2) Use ϵ-greedy policy

            * ϵ-greedy is a family of soft-policy.
              ϵ-greedy is a soft-policy that is close to greedy.


- Monte Carlo Control

      # Policy improvement step #

      Policy improvement를 위해서는 given policy에 대한 정확한 q 값이 필요.
      그리고 다시 말하지만, 정확한 q를 위해서는 아래의 두 방법을 사용한다.

      1) Use greedy policy + exploring starts (ES)
      2) Use ϵ-greedy policy
    
      이를 통해 정확하게 추정된 q가 있어야 optimal policy를 찾아낼 수 있다.

      * greedy policy 뿐만 아니라, ϵ-greedy policy도 monotonic improvment가 보장된다.

---

- Off-policy prediction


      # problem 1)
  
      q value를 기반으로 policy improvement를 하려면 exploration을 해야한다.
      따라서, ES를 이용하는데, 이게 일반적으로 적용하기 힘든 방법이다.
      (Random 리스폰을 기반으로 하기에, 환경과 actual intraction을 한다면 적용이 힘듦)

      ➔ 이에 대한 대안으로, soft greedy policy family 를 이용할 수 있다.

      # problem 2)
  
      우리는 optimal한 행동들을 기반으로한 q value를 얻고 싶다.
      근데, exploration을 해야하니, non-optimal한 행동들도 섞어서 하고 있다.
  
      즉,
      1) on policy Monte Carlo + greedy policy + ES
      2) on policy Monte Carlo + ϵ-greedy policy
  
      를 이용하는 방법들은 optimal policy가 아니라, exploration을 계속 하는 'near-optimal policy'를 구한 것이다..!
  
      그러면, exploration은 exploration 대로 하면서, q value는 optimal한 행동들을 기반으로 추정할 수 없을까?
      ➔ 'off-policy' 도입!

      * On-policy vs Off-policy

          On-policy  : Fast convergence and simple
          Off-policy : Slow convergence and high-variance ➔ But powerful! 
          (e.g., learn from data of expert policy)
    
      * 'Assumption of coverage' in off-policy learning

          In order to use episodes from behavior policy to estiamte values under the target policy,
          we required that every action taken under target policy is also taken under behavior policy.
          즉, 적어도 behavior policy가 target policy가 고르는 행동들을 훑어야한다는 얘기. 

      * Off policy를 위한 Importance sampling 종류도 몇 가지 있다.

        1) ordinary importance sampling (varince 有)
        2) weighted importance sampling (bias 有) ➔ preferred!

        + 3) discounting-aware importance sampling
           Gt의 internal structure level에서 off policy의 variance를 줄이는 방법
           ➔ discount factor γ 이용 
  
        + 4) per-decision importance sampling
           discount factor γ 를 사용하지 않고도 off policy의 variance를 줄일 수 있는 방법


  

      
          



    

            
