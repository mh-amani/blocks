# inspired by hugginface's get_polynomial_decay_schedule_with_warmup
from .abstract_scheduler import AbstractScheduler

class LinearScheduler(AbstractScheduler):
    def __init__(self,
                 num_warmup_steps: int,
                 num_training_steps: int,
                 hp_init: float,
                 hp_end: float,
                 power: float =1.0,
                ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.hp_init = hp_init
        self.hp_end = hp_end
        self.power = power
        
        self.annealing = self.hp_init >= self.hp_end
        
    
    def update(self,current_hyperparameter_val, current_step):
        current_step = current_step + 1 #update done at end of epoch so increment by one
        #case where we want to anneal our hyperparameter throughout training
        if self.annealing:
            
            if current_step < self.num_warmup_steps:
                return current_hyperparameter_val#self.hp_init * float(current_step) / float(max(1, self.num_warmup_steps))
            
            elif current_step > self.num_training_steps:
                return self.hp_end  
            
            else:
                hp_range = self.hp_init - self.hp_end 
                decay_steps = self.num_training_steps - self.num_warmup_steps
                pct_remaining = 1 - (current_step - self.num_warmup_steps) / decay_steps
                decay = hp_range * pct_remaining**self.power + self.hp_end
                return decay
            
        #case where we want to increase the value of our hyperparameter throughout training (e.g entmax)
        else:
            
            if current_step < self.num_warmup_steps:
                return current_hyperparameter_val#self.hp_init * (1 - float(current_step) / float(max(1, self.num_warmup_steps)))
            
            elif current_step > self.num_training_steps:
                return self.hp_end  
            
            else:
                hp_range =  self.hp_end - self.hp_init
                decay_steps = self.num_training_steps - self.num_warmup_steps
                pct_remaining = float((current_step - self.num_warmup_steps)) / decay_steps
                growth = hp_range * pct_remaining**self.power + self.hp_init
                return growth
        