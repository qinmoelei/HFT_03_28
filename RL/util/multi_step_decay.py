def get_ada(ada, decay_freq=2, ada_counter=0, decay_coffient=0.5):
    #decay auxilary task coffient
    if ada_counter % decay_freq == 1:
        ada = decay_coffient * ada
    return ada


def get_epsilon(epsilon,
                max_epsilon=1,
                epsilon_counter=0,
                decay_freq=2,
                decay_coffient=0.5):
    #decay epsilon
    if epsilon_counter % decay_freq == 1:
        epsilon = epsilon + (max_epsilon - epsilon) * decay_coffient
    return epsilon


def get_beta(beta_init, epoch_number, total_epoch):
    return beta_init + (1 - beta_init) * (epoch_number / total_epoch)
