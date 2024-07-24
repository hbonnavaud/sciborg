

def copy_weights(module_to_modify, module_to_copy, tau=0.1):
    for param_1, param_2 in zip(module_to_modify.parameters(), module_to_copy.parameters()):
        param_1.data.copy_(
            param_1.data * (1.0 - tau) + param_2.data * tau
        )
    return module_to_modify

    # for name, param in module_to_modify.state_dict().items():
    #     if name in module_to_copy.state_dict():
    #         param.copy_(param * (1.0 - tau) + other_nn.state_dict()[name] * tau)
