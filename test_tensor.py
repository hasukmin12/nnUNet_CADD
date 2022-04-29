import torch

# T = torch.tensor([1,2,3,4,5])
# print(T)
# i = 2
# T = torch.cat([T[0:i], T[i+1:]])
# print(T)



x = torch.arange(10)
print(x)
value = 5

x = x[x!=value]
print(x)





imaginated_values = value_model(flatten_imaginated_states,
                                flatten_imaginated_gru_hiddens).view(imagination_horizon + 1, -1)

## compute lambda target
lambda_target_values = utils.lambda_target(imaginated_rewards, imaginated_values,
                                           gamma, lambda_)

## update_value model



value_loss = 0.5 * mse_loss(imaginated_values, lambda_target_values.detach())
value_optimizer.zero_grad()
value_loss.backward(retain_graph=True)
# --------------------------------------------------------------
# retain_graph=True :
# reserve the parameters in the process of backpropagation
# when encountering two more outputs from the forward processing.
# --------------------------------------------------------------
clip_grad_norm_(value_model.parameters(), clip_grad_norm)
value_optimizer.step()

## update action model (multiply -1 for gradient ascent)
action_loss = -1 * (lambda_target_values.mean())
action_optimizer.zero_grad()
action_loss.backward()
clip_grad_norm_(action_model.parameters(), clip_grad_norm)
action_optimizer.step()