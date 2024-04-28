#### Curiosity-dirven exploration

----

- Brief summary

        # 1 line summary
        ➜ This paper proposed a curiosity-driven 'intrinsic' reward generating mechanism that scales to high-dimensional visual inputs.

        # concept
        Raw sensory image level의 prediction이 아니라,
        agent가 수행한 action에 의해 변화할 수 있는 것들에만 관련된 정보들이 있는 feature space level의 prediction 수행
        ➜ Forward Dynamics Model 이용

          Feature space는 self-supervision으로 구축
          ➜ Inverse Dynamics Model 이용

        Forward Dynamics Model의 prediction error에 기반해서 curiosity 계산하고, 이를 통해 intrinsic reward signal 생성.

        * Intrinsic reward is important, especially when the extrinsic rewards are sparse.



