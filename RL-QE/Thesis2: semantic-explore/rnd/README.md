#### Random Network Distillation (RND)

---


- Brief summary 


        * Key factors

            1) Define 2 networks (used for making intrinsic reward)
                ➔ target network : fixed and random initialized NN 
                    ➔ make prediction targets for predictor network
    
                ➔ predictor network : predict targets from target network, by minimizing the errors
                    ➔ "prediction error" == "intrinsic reward"!


            2) Define 2 seperate value functions (one for intrinsic reward and the other for extrinsic reward)
                ➔ V-E : value function of extrinsic reward (episodic setting works better)
                ➔ V-I : value function of intrinsic reward (non-episodic setting works better)

                    ➔ intrinsic rewards만 가지고서 exploration할 때, non-episodic으로 setting하는게 더 많은 exploration을 한다.
                    ➔ intrinsic + extrinsic rewards로 exploration할 때는, episodic extrinsic reward & non-episodic intrinsic reward 써라.

                Use 2 head value function V := (V-E) + (V-I) works better. 
            

                * extrinsic reward : stationary
                * intrinsic reward : non-stationary (it results in POMDP)
                

