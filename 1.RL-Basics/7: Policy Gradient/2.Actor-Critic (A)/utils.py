beta  = 0.005

def soft_update(Target_Network, Current_Network):
    for target_param, current_param in zip(Target_Network.parameters(), Current_Network.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - beta) + current_param.data * beta)
