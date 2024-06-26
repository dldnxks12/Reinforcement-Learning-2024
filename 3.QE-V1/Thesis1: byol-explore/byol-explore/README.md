#### BYOL-Explore

---

- Abstract


            BYOL-Explore ?
                ➔ curiosity-driven exploration method for visually-complex Env.
                ➔ Simple and High Performance. 

            BYOL-Explore learns ?
                ➔ World representation (Representation Learning : BYOL)
                ➔ World dynamics (World Model)
                ➔ Exploration policy (Curiosity-driven exploration)

                by optimizing a 'single' loss.

                BYOL-E learns a World Model's representation based on a self-supervised prediction loss.
                BYOL-E trains a curiosity-driven policy with the 'same' loss.

                That is... BYOL-E uses the world model's loss to drive exploration!
                ➔ we can solve both the problems of building world model and curiosity-driven policy with a single objective function!!



            * Experiment on DM-HARD-8 tasks
                ➔ need efficient exploration
                    ➔ in order to reach the goal and get the reward, 
                    they require completing a sequence of prcise, orderly interactions with the physical objects in the Env. 
                


            Additional method used for solving visually-complex Env.
                ➔ extrinsic reward + BYOL's intrinsic reward.

                ➔ Intrinsic reward
                    ➔ made from residuals between World Model's prediction and real data

            
            * Three types of reward
                [1] Extrinsic reward
                    ➔ Externally provided rewards

                [2] Intrinsic reward
                    ➔ Rewards generated by the agent themselves,
                      based on their internal state and their own model of the Env.

                [3] Shaped reward 
                    ➔ Combination of extrinsic and intrinsic rewards
    


- Introduction


        Visually-complex Env. 와 같이 풍부한 정보들이 있는 환경에서는 어느 쪽에 가서 탐험하는게 더 재밋을지...
        에이전트 스스로 결정하기가 쉽지 않다 ➔ Curiosity-driven exploration 방법 등장!

            * Curiosity-driven exploration
            [1] learn the 'World Model' (i.e. learn the 'predictive model' of some information about the world)
            [2] using errors between the predictions of World Model and the real-experiences to build 'intrisic rewards'

            ➔ 이 intrinsic reward를 최적화하려는 RL agent는 World Model이 부정확하게 예측한 부분으로 유도한다.    
                ➔ World Model이 향상/수정/보완될 수 있는 trajectory를 생성한다!
                ➔ 다시 말해, World Model의 특성이 exploration 의 퀄리티에 크게 영향을 준다.
                ➔ 따라서, World Model 학습과 exploration policy의 학습이 각각 별개의 문제가 아니고, 하나의 문제가 된다!


        BYOL learns a World Model with a self-supervised prediction loss.
        BYOL uses same loss to train a curiosity-driven policy.
            ➔ Thus, BYOL uses a 'single learning objective' in learning 'both' World Model and curiosity-driven policy..!

        i.e., BYOL uses the loss in learning World Model, to drive exploration!
        (BYOL not only learns the World Model, but also uses the loss to drive exploration)


        저자의 주장 BYOL? 간단하고, 일반화가 잘되고, 성능이 좋다.

            Sparse reward를 갖는 복잡한 vision 환경에서 BYOL은 다른 유명한 curiosity-driven exploration method 능가.
                ➔ BYOL >> Random Network Distillation (RND) 
                ➔ BOYL >> Intrinsic Curiosity Module (ICM)


---

- Method


    The agent has three components:

    [1] A self-supervised latent-predictive world model (BYOL-Explore)
    [2] A generic reward normalization and prioritization scheme
    [3] An off-the-shelf RL agent, optionally sharing its own representation with BYOL-Explore's world model.

    ----------------------------------------------------------------------------------------------------------

    [1] Latent-Predictive World Model

    BYOL-Explore world model is a multi-step predictive world model, operating a the latent level.
    

    