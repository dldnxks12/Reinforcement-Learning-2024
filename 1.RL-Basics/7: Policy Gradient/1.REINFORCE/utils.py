import gymnasium as gym
import torch
import math

def gen_epsiode(env, device, policy):
  states , actions, rewards = [], [], []

  state = env.reset()[0]
  done  = False
  score = 0

  while not done:

    state = torch.FloatTensor(state).to(device)

    # TODO : action에 대한 Softmax 분포 return
    probabilities = policy(state)

    # TODO : return된 분포에 따라 action 선택 (discrete action)
    action = torch.multinomial(probabilities, 1).item() # 반환받은 분포에서 1개를 뽑는다. (Index 반환)
    next_state, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        done   = True

    states.append(state)
    actions.append(action)
    rewards.append(reward)

    score += reward

    state = next_state

  return states, actions, rewards, score


gamma = 0.99
def G(rewards):
  G_0 = 0
  for i in range(len(rewards) - 1):
    gam = math.pow(gamma, i) # gamma의 i제곱
    G_0 += gam*rewards[i]
  return G_0