class AbstractScheduler:
    def update(self,current_hyperparameter_val, global_steps):
        raise NotImplementedError()