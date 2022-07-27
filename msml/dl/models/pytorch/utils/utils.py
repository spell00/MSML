import torch


def get_optimizer(model, learning_rate, weight_decay, optimizer_type, momentum=0.9):
    """

    Args:
        model: The PyTorch model to optimize
        learning_rate: The optimizer's learning rate
        weight_decay: The optimizer's weight decay
        optimizer_type: The optimizer's type [adam or sgd]
        momentum:

    Returns:

    """
    # TODO Add more optimizers
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay,
                                     )
    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay,
                                    momentum=momentum
                                    )
    return optimizer


