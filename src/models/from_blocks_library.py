import hydra
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Subset
import collections.abc
import src.metrics
import os
import jsonlines
import json
import torch
from omegaconf import OmegaConf
from src.utils.metrics import pad_label_label
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchmetrics.wrappers import BootStrapper
import torchmetrics
import code
import gc
# from memory_profiler import profile

class XZAutoencoder(LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.save_hyperparameters(ignore=["datamodule",])
                
        # if loading a pretrained model, but need to change some of the parameters
        if self.hparams.get('substitute_config'):
            self._update_params(self.hparams, self.hparams.substitute_config)
        
        self.pad_token_id = self.hparams.special_tokens.index('[pad]')        
        self.automatic_optimization = False

        # self.loss = torch.nn.NLLLoss(ignore_index=self.pad_token_id)
        self.loss = CrossEntropyLoss(ignore_index=self.pad_token_id, label_smoothing=0.01)
        self.loss_coeff = self.hparams.model_params.loss_coeff
        self.usexz = self.hparams.model_params['usexz']
        self.usez = self.hparams.model_params['usez']
        self.usex = self.hparams.model_params['usex']

        self.batch_size = self.hparams.dataset_parameters.batch_size

        self.acc_grad_batch = self.hparams.model_params.acc_grad_batch
        assert self.acc_grad_batch > 0, "acc_grad_batch must be greater than 0"

        # collate_fn
        train_dataset = kwargs['datamodule'].data_train
        self.pretokenized_flag = 0
        self.collator = hydra.utils.instantiate(self.hparams.collator, train_dataset, special_tokens=self.hparams.special_tokens, _recursive_ = False)
        
        # Model
        disc_dims_x, disc_dims_z = self.discretizer_dimensions()
        self.disc_x, self.disc_z, self.model_x_to_z, self.model_z_to_x, self.model_x_to_z_to_x, self.model_z_to_x_to_z = \
                hydra.utils.instantiate(self.hparams.blocks_model, disc_dims_x=disc_dims_x, disc_dims_z=disc_dims_z, _recursive_=False) 

        
        self.acc_mask= {'x': None, 'z': None}
        numclasses = {'X': self.collator.tokenizer_x.get_vocab_size(), 'Z': self.collator.tokenizer_z.get_vocab_size()}
        self.accuracy = torch.nn.ModuleDict()
        self.accuracy_sentence = torch.nn.ModuleDict()
        self.manual_accuracy = {}
        self.manual_accuracy_sentence = {}
        for stage in ['val', 'test']:
            for mode in ['teacherforced', 'autoreg', 'autoreg_hidden_layer']:
                for variable in ['X', 'Z']:
                    acc_name = f'{stage}/{mode}/accuracy/{variable}'
                    sentence_acc_name = f'{stage}/{mode}/sentence-accuracy/{variable}'
                    # self.accuracy[acc_name] = src.metrics.Accuracy(self.pad_token_id)
                    self.accuracy[acc_name] = torchmetrics.classification.MulticlassAccuracy(num_classes=numclasses[variable] ,ignore_index=self.pad_token_id)
                    self.accuracy_sentence[sentence_acc_name] = torchmetrics.classification.MulticlassExactMatch(num_classes=numclasses[variable] ,ignore_index=self.pad_token_id)
                    self.manual_accuracy[acc_name] = {'correct': 0, 'total': 0}
                    self.manual_accuracy_sentence[sentence_acc_name] = {'correct': 0, 'total': 0}

        self.wrong_x_predictions = []
        self.wrong_z_predictions = []

        self.log_gradient_stats = self.hparams.model_params.log_gradient_stats
        self.num_steps_log_gradient_stats = self.hparams.model_params.num_steps_log_gradient_stats
        self.log_gradient_stats_batch_size = self.hparams.model_params.log_gradient_stats_batch_size

        self.aggregated_grads = {}
                    
    def setup(self, stage: str) -> None:
        
        if self.log_gradient_stats and not hasattr(self, 'log_gradient_dataloader'):
            indices = range(self.log_gradient_stats_batch_size * self.num_steps_log_gradient_stats)
            self.log_gradient_dataset = Subset(self.trainer.datamodule.data_train, indices)
            self.log_gradient_dataloader = DataLoader(
                self.log_gradient_dataset,
                batch_size=self.log_gradient_stats_batch_size,
                shuffle=False,
                collate_fn=self.collator.collate_fn
            )
    

    def forward(self, batch, stage='train'):
        data_type = batch['data_type']
        code.interact(local=locals())
        x_ids = batch['x_ids']
        z_ids = batch['z_ids']

        self.log(f"{stage}/x_data_available", float(data_type[0]), batch_size=self.batch_size, sync_dist=True)
        self.log(f"{stage}/z_data_available", float(data_type[1]), batch_size=self.batch_size, sync_dist=True)
        self.log('global_step', float(self.global_step), batch_size=self.batch_size, sync_dist=True)
        
        outputs = {}
        outputs['supervised_seperated'] = None
        outputs['xzx'] = None
        outputs['zxz'] = None

        losses = {}
        losses['supervised_seperated'] = None
        losses['xzx'] = None
        losses['zxz'] = None
        losses['supervised_seperated_x'] = None
        losses['supervised_seperated_z'] = None
        losses['quantization_supervised_seperated'] = None
        losses['quantization_xzx'] = None
        losses['quantization_zxz'] = None
        
        # Supervision on Z and Supervision on X seperately
        if (data_type[0] and data_type[1] and self.usexz) or stage!='train':
            # print('xz, zx')
            # print(torch.cuda.memory_allocated() / 1024**2)
            # print(torch.cuda.memory_reserved() / 1024**2)
            output_supervised_seperated_z = self.model_x_to_z(input_ids=x_ids, output_ids=z_ids, teacher_force_output=True)
            output_supervised_seperated_x = self.model_z_to_x(input_ids=z_ids, output_ids=x_ids, teacher_force_output=True)
            # print(torch.cuda.memory_allocated() / 1024**2)
            # print(torch.cuda.memory_reserved() / 1024**2)
            loss_x = self.loss((output_supervised_seperated_x['logit'][:, :-1]).permute(0, 2, 1), x_ids[:, 1:])
            loss_z = self.loss((output_supervised_seperated_z['logit'][:, :-1]).permute(0, 2, 1), z_ids[:, 1:])

            loss_supervised_seperated = self.loss_coeff['supervised_seperated_x'] * loss_x + self.loss_coeff['supervised_seperated_z'] * loss_z
            outputs['supervised_seperated'] = {'output_z': output_supervised_seperated_z, 'output_x': output_supervised_seperated_x}
            losses['supervised_seperated'] = loss_supervised_seperated
            losses['supervised_seperated_x'] =  loss_x
            losses['supervised_seperated_z'] = loss_z
            losses['quantization_supervised_seperated'] = output_supervised_seperated_x['quantization_loss'] + output_supervised_seperated_z['quantization_loss']

        # Unsupervized xzx pass
        if (data_type[0] and not data_type[1]) or (stage!='train') or (data_type[0] and data_type[1] and self.usex):
            # print('xzx')
            # print(torch.cuda.memory_allocated() / 1024**2)
            # print(torch.cuda.memory_reserved() / 1024**2)
            output_xzx = self.model_x_to_z_to_x(x_ids=x_ids, max_y_length=self.hparams.model_params.max_z_length, 
                                                z_ids=x_ids, teacher_force_z=True)
            # print(torch.cuda.memory_allocated() / 1024**2)
            # print(torch.cuda.memory_reserved() / 1024**2)
            loss_xzx = self.loss(torch.nn.LogSoftmax(dim=-1)(output_xzx['logit_z'][:, :-1]).permute(0, 2, 1), x_ids[:, 1:])
            outputs['xzx'] = output_xzx
            losses['xzx'] = loss_xzx  
            losses['quantization_xzx'] = output_xzx['quantization_loss'] 
        
        # Unsupervized zxz pass
        if (data_type[1] and not data_type[0]) or (stage!='train') or (data_type[0] and data_type[1] and self.usez):
            # print('zxz')
            # print(torch.cuda.memory_allocated() / 1024**2)
            # print(torch.cuda.memory_reserved() / 1024**2)
            output_zxz = self.model_z_to_x_to_z(x_ids=z_ids, max_y_length=self.hparams.model_params.max_x_length,
                                                 z_ids=z_ids, teacher_force_z=True)
            # print(torch.cuda.memory_allocated() / 1024**2)
            # print(torch.cuda.memory_reserved() / 1024**2)
            loss_zxz = self.loss(torch.nn.LogSoftmax(dim=-1)(output_zxz['logit_z'][:, :-1]).permute(0, 2, 1), z_ids[:, 1:])
            outputs['zxz'] = output_zxz
            losses['zxz'] = loss_zxz
            losses['quantization_zxz'] = output_zxz['quantization_loss']
      
        loss = 0 
        for key in losses:
            if losses[key] is not None:
                losses[key] = losses[key].mean()
                self.log(f'{stage}/loss/{key}', losses[key], batch_size=self.batch_size, sync_dist=True)
                if self.loss_coeff.get(key) is not None:
                    loss += (self.loss_coeff[key]>0) * self.loss_coeff[key] * losses[key]  

        self.log(name=f'{stage}/loss', value=loss, batch_size=self.batch_size, prog_bar=True, sync_dist=True)  

        return loss, losses, outputs       
    
            
    def training_step(self, batch, batch_idx):
        if self.hparams.model_params.get('use_pc_grad', False):
            self.pc_grad_update(batch, batch_idx)
        else:
            self.gd_update(batch, batch_idx)

    
    def gd_update(self, batch, batch_idx):
        loss, losses, outputs = self.forward(batch)
        loss = loss / self.acc_grad_batch * 1.0
        self.manual_backward(loss)

        if batch_idx == 0:
            log_string = f"---------------------------------------------\nEpoch: {self.trainer.current_epoch}\n"
            name_list = ['model_z_to_x.decoder.layernorm_embedding.bias', 'model_z_to_x.decoder.layernorm_embedding.weight', 
                        'model_z_to_x.decoder.layers.7.final_layer_norm.bias', 'model_z_to_x.decoder.layers.7.final_layer_norm.weight',
                        'model_x_to_z.decoder.layernorm_embedding.bias', 'model_x_to_z.decoder.layernorm_embedding.weight',
                        'model_x_to_z.decoder.layers.7.final_layer_norm.bias', 'model_x_to_z.decoder.layers.7.final_layer_norm.weight']
            
            for name, param in iter(self.named_parameters()):
                if param._grad is not None and (name.startswith('disc') or name in name_list):
                    f = '{: <75}'.format(name) + '{: <10}'.format(str(param._grad.abs().mean().cpu().numpy().round(decimals=4))) +  '     '  + '{: <10}'.format(str(param.abs().mean().detach().cpu().numpy().round(decimals=4))) + '\n'
                    log_string += f

            # Specify the path of the text file
            file_path = "param_grad_log.txt"

            # Open the file in append mode and write the log_string
            with open(file_path, "a") as file:
                file.write(log_string)


        if (batch_idx + 1) % self.acc_grad_batch == 0:
            optimizers = self.optimizers()

            for optimizer in optimizers:
                # Check if any optimizer parameter is NaN
                self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                optimizer.step()
                optimizer.zero_grad()

        with torch.no_grad():
            for name, param in iter(self.named_parameters()):
                # if name.startswith('disc') and param.requires_grad:
                param.clamp_(-1, 1)

        return loss

    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        for id, scheduler in enumerate(schedulers):
            # If the selected scheduler is a ReduceLROnPlateau scheduler.
            scheduler.step(self.trainer.callback_metrics[self.hparams.lr_scheduler.monitor])
            self.log(name=f'lr-scheduler/{self.module_names[id]}', value=scheduler._last_lr[0], batch_size=self.batch_size, sync_dist=True)

    def compute_accuracy_measures(self, batch, batch_idx, stage):
        assert np.all(batch['data_type']), "compute_accuracy_measures: data_type must be supervised"

        _, _, outputs = self.forward(batch, stage='val')

        accuracies = {}

        x_ids = batch['x_ids'].detach()
        z_ids = batch['z_ids'].detach()
        x_pad_mask = torch.logical_not(torch.eq(x_ids, self.pad_token_id))
        z_pad_mask = torch.logical_not(torch.eq(z_ids, self.pad_token_id))
        
        teacher_forced_available = outputs['supervised_seperated'] is not None
        autoreg_z_available = outputs['zxz'] is not None
        autoreg_x_available = outputs['xzx'] is not None

        if teacher_forced_available:
            x_hat_ids_teacherforced = outputs['supervised_seperated']['output_x']['id'][:, :-1].detach()
            x_hat_ids_teacherforced = x_hat_ids_teacherforced * x_pad_mask[:, 1:]
            z_hat_ids_teacherforced = outputs['supervised_seperated']['output_z']['id'][:, :-1].detach()
            z_hat_ids_teacherforced = z_hat_ids_teacherforced * z_pad_mask[:, 1:]

            x_ids_teacherforced, x_hat_ids_teacherforced = pad_label_label(x_ids[:, 1:], x_hat_ids_teacherforced, self.pad_token_id)
            z_ids_teacherforced, z_hat_ids_teacherforced = pad_label_label(z_ids[:, 1:], z_hat_ids_teacherforced, self.pad_token_id)
            
            self.accuracy_measures(x_ids_teacherforced, x_hat_ids_teacherforced, stage, 'X', 'teacherforced')
            self.accuracy_measures(z_ids_teacherforced, z_hat_ids_teacherforced, stage, 'Z', 'teacherforced')
        
        if autoreg_z_available:
            z_hat_ids_autoreg = outputs['zxz']['id_z'][:, :-1].detach()
            x_hat_ids_autoreg = outputs['zxz']['id_y'].detach()

            z_hat_ids_autoreg = z_hat_ids_autoreg * z_pad_mask[:, 1:]
            
            x_ids_autoreg, x_hat_ids_autoreg = pad_label_label(x_ids[:, 1:], x_hat_ids_autoreg, self.pad_token_id)
            x_ids_autoreg_mask = torch.logical_not(torch.eq(x_ids_autoreg, self.pad_token_id))
            x_hat_ids_autoreg = x_hat_ids_autoreg * x_ids_autoreg_mask

            
            self.accuracy_measures(z_ids[:, 1:], z_hat_ids_autoreg, stage, 'Z', 'autoreg')
            self.accuracy_measures(x_ids_autoreg, x_hat_ids_autoreg, stage, 'X', 'autoreg_hidden_layer')

         
        if autoreg_x_available:
            x_hat_ids_autoreg = outputs['xzx']['id_z'][:, :-1].detach()
            z_hat_ids_autoreg = outputs['xzx']['id_y'].detach()
            
            x_hat_ids_autoreg = x_hat_ids_autoreg * x_pad_mask[:, 1:]

            z_ids_autoreg, z_hat_ids_autoreg = pad_label_label(z_ids[: ,1:], z_hat_ids_autoreg, self.pad_token_id)
            z_ids_autoreg_mask = torch.logical_not(torch.eq(z_ids_autoreg, self.pad_token_id))
            z_hat_ids_autoreg = z_hat_ids_autoreg * z_ids_autoreg_mask

            
            self.accuracy_measures(x_ids[:, 1:], x_hat_ids_autoreg, stage, 'X', 'autoreg')
            self.accuracy_measures(z_ids_autoreg, z_hat_ids_autoreg, stage, 'Z', 'autoreg_hidden_layer')

        return outputs


    def accuracy_measures(self, ids, hat_ids, stage, variable, type, log=True):
        
        # shifting to make the sequences aligned, removing bos
        acc_device = self.device
        ids = ids.to(acc_device)
        hat_ids = hat_ids.to(acc_device)

        pad_mask = torch.logical_not(torch.eq(ids, self.pad_token_id))

        acc_name = f'{stage}/{type}/accuracy/{variable}'
        sentence_acc_name = f'{stage}/{type}/sentence-accuracy/{variable}'
        
        self.manual_accuracy[acc_name]['correct'] += torch.sum(torch.eq(hat_ids, ids)).cpu().numpy() - torch.sum(torch.logical_not(pad_mask)).cpu().numpy()
        self.manual_accuracy[acc_name]['total'] += torch.sum(pad_mask).cpu().numpy()
        
        self.manual_accuracy_sentence[sentence_acc_name]['correct'] += torch.sum(torch.eq(hat_ids, ids).all(axis=-1)).cpu().numpy()
        self.manual_accuracy_sentence[sentence_acc_name]['total'] += len(ids)
        

        self.accuracy[acc_name].update(hat_ids.reshape(-1), ids.reshape(-1))
        self.accuracy_sentence[sentence_acc_name].update(hat_ids, ids)
        self.log(acc_name, self.accuracy[acc_name], batch_size=self.batch_size, sync_dist=True)
        self.log(sentence_acc_name, self.accuracy_sentence[sentence_acc_name], batch_size=self.batch_size, sync_dist=True)

        if sentence_acc_name.startswith('val/autoreg_hidden_layer/sentence-accuracy/X'):
            wrong_prediction = torch.where(torch.logical_not(torch.eq(hat_ids, ids).all(axis=-1)))
            self.wrong_x_predictions = self.wrong_x_predictions + wrong_prediction[0].cpu().numpy().tolist()
        
        if sentence_acc_name.startswith('val/autoreg_hidden_layer/sentence-accuracy/Z'):
            wrong_prediction = torch.where(torch.logical_not(torch.eq(hat_ids, ids).all(axis=-1)))
            self.wrong_z_predictions = self.wrong_z_predictions + wrong_prediction[0].cpu().numpy().tolist()
    
    def calculate_gradient_stats(self, gradient_dict):
        cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        similarities = {}
        grad_norm_means = {}
        grad_norm_stds = {}

        # Get the intersection of parameter names across all losses
        shared_params = set.intersection(*(set(grad_dict.keys()) for grad_dict in gradient_dict.values()))

        for loss1 in ['supervised_seperated_x', 'supervised_seperated_z', 'zxz', 'xzx']:
            grad_rms = torch.stack([torch.sqrt(torch.mean(torch.square(gradient_dict[loss1][param_name]))) for param_name in gradient_dict[loss1].keys()])
            grad_norm_means[loss1] = torch.mean(grad_rms)
            grad_norm_stds[loss1] = torch.std(grad_rms)
        for loss1 in ['zxz', 'xzx']:
            for loss2 in ['supervised_seperated_x', 'supervised_seperated_z']:
                    # Compute cosine similarity for each shared parameter and average
                    similarities[f'{loss1}-{loss2}'] = torch.mean(torch.stack(
                        [cosine_similarity(gradient_dict[loss1][param_name], gradient_dict[loss2][param_name])
                         for param_name in shared_params]
                    ))
        return {'cosine_similarities': similarities, 'grad_norm_rms': grad_norm_means, 'grad_norm_stds_across_parameter': grad_norm_stds}


    def validation_step(self, batch, batch_idx):
        self.compute_accuracy_measures(batch, batch_idx, stage='val')        
    

    # @profile
    def on_validation_epoch_end(self):
        # self.correct_predictions_mask()
        for key in self.manual_accuracy:
            if self.manual_accuracy[key]['total'] > 0:   
                self.log('manual/' + key, self.manual_accuracy[key]['correct']/self.manual_accuracy[key]['total'], batch_size=self.batch_size, sync_dist=True)
                # if key.startswith('val/autoreg_hidden_layer'):
                    # print('manual/' + key, self.manual_accuracy[key]['correct']/self.manual_accuracy[key]['total'])
            self.manual_accuracy[key] = {'correct': 0, 'total': 0}
        for key in self.manual_accuracy_sentence:
            if self.manual_accuracy_sentence[key]['total'] > 0:
                self.log('manual/' + key, self.manual_accuracy_sentence[key]['correct']/self.manual_accuracy_sentence[key]['total'], batch_size=self.batch_size, sync_dist=True)
                # if key.startswith('val/autoreg_hidden_layer'):
                    # print('manual/' + key, self.manual_accuracy_sentence[key]['correct']/self.manual_accuracy_sentence[key]['total'])
            self.manual_accuracy_sentence[key] = {'correct': 0, 'total': 0}

         # if self.validation_step_gradient_logging:
        if self.log_gradient_stats:
            self.train()
            # Ensure gradient computation is enabled
            torch.set_grad_enabled(True)

            for batch in self.log_gradient_dataloader:
                # sending to device
                # for key in batch:
                #     if isinstance(batch[key], torch.Tensor):
                #         batch[key] = batch[key].to(self.device)
                batch = self.trainer.datamodule.transfer_batch_to_device(batch, device=self.device, dataloader_idx=0)
                # Forward pass to compute losses
                _, losses, _ = self.forward(batch, stage='val')
            
                # Calculate gradients for each loss separately and store them
                gradient_dict = {}
                for loss_name in ['supervised_seperated_x', 'supervised_seperated_z', 'zxz', 'xzx']:
                    if losses[loss_name] is not None:
                        self.manual_backward(losses[loss_name], retain_graph=(loss_name != 'xzx'))
                        gradient_dict[loss_name] = {name: param.grad for name, param in self.named_parameters() if param.grad is not None}
                        # Zero out gradients after each backward pass
                        self.zero_grad()
                        torch.cuda.empty_cache()
                        gc.collect()
                        

                # Calculate cosine similarity between gradients of different losses
                grad_stat = self.calculate_gradient_stats(gradient_dict)
                
                del gradient_dict
                if 'grad_stats' not in locals():
                    grad_stats = {key: {subkey: [grad_stat[key][subkey]] for subkey in grad_stat[key].keys()} for key in grad_stat.keys()}
                else:
                    for key in grad_stats.keys():
                        for subkey in grad_stats[key].keys():
                            grad_stats[key][subkey].append(grad_stat[key][subkey])
                del grad_stat
                torch.cuda.empty_cache()
                gc.collect()

            # log mean and std of cosine similarity and gradient norms
            for key in grad_stats.keys():
                for subkey in grad_stats[key].keys():
                    grad_stats[key][subkey] = torch.stack(grad_stats[key][subkey])
                    self.log(f'gradient_stats/{key}/{subkey}/mean', torch.mean(grad_stats[key][subkey]), batch_size=self.batch_size, sync_dist=True)
                    self.log(f'gradient_stats/{key}/{subkey}/std', torch.std(grad_stats[key][subkey]), batch_size=self.batch_size, sync_dist=True)
            
            # Disable gradient computation as it's typically not needed after this
            torch.set_grad_enabled(False)
            self.eval()
        

        # add epoch number, and cosine similarity of x and z embedding vectors to a text file.
        # Create a string to write to the text file
        # disc_x_cosine_sim = self.format_matrix(self.dictionary_cosine_sim('x'))
        # disc_z_cosine_sim = self.format_matrix(self.dictionary_cosine_sim('z'))
        # disc_x_inner_prod = self.format_matrix(self.dictionary_inner_prod_sim('x'))
        # disc_z_inner_prod = self.format_matrix(self.dictionary_inner_prod_sim('z'))
        # log_string = f"Epoch: {self.trainer.current_epoch}\nDisc_x_cosine_sim:\n{disc_x_cosine_sim}\nDisc_z_cosine_sim:\n{disc_z_cosine_sim}\nDisc_x_inner_prod:\n{disc_x_inner_prod}\nDisc_z_inner_prod:\n{disc_z_inner_prod}\n"

        # # Specify the path of the text file
        # file_path = "disc_logs.txt"

        # # Open the file in append mode and write the log_string
        # with open(file_path, "a") as file:
        #     file.write(log_string)

    def test_step(self, batch, batch_idx):
        outputs = self.compute_accuracy_measures(batch, batch_idx, stage='test')

    
    def on_test_epoch_end(self):
        # self.correct_predictions_mask()
        for key in self.manual_accuracy:
            if self.manual_accuracy[key]['total'] > 0:   
                self.log('manual/' + key, self.manual_accuracy[key]['correct']/self.manual_accuracy[key]['total'], batch_size=self.batch_size, sync_dist=True)
            self.manual_accuracy[key] = {'correct': 0, 'total': 0}
        for key in self.manual_accuracy_sentence:
            if self.manual_accuracy_sentence[key]['total'] > 0:
                self.log('manual/' + key, self.manual_accuracy_sentence[key]['correct']/self.manual_accuracy_sentence[key]['total'], batch_size=self.batch_size, sync_dist=True)
            self.manual_accuracy_sentence[key] = {'correct': 0, 'total': 0}
    

    def correct_predictions_mask(self, dataloader=None):
        correct_z_ids = 0
        correct_x_ids = 0
        total_z_ids = 0
        total_x_ids = 0
        correct_x_sentence_ids = 0
        correct_z_sentence_ids = 0

        self.eval()
        if dataloader is None:
            dataloader = self.trainer.datamodule.val_dataloader()
        for batch in dataloader:
            
            collated_batch = batch
            collated_batch_device = self.trainer.datamodule.transfer_batch_to_device(batch=collated_batch, device=self.device, dataloader_idx=0)

            loss, losses, outputs = self.forward(batch=collated_batch_device, stage='val')

            x_ids = collated_batch_device['x_ids']
            x_hat_ids = outputs['zxz']['x_hat_ids']
            x_ids, x_hat_ids = pad_label_label(collated_batch_device['x_ids'][:, 1:], x_hat_ids, pad_token_id=self.pad_token_id)
            x_pad_mask = torch.logical_not(torch.eq(x_ids, self.pad_token_id))
            x_hat_ids = x_hat_ids * x_pad_mask

            z_ids = collated_batch_device['z_ids']
            z_hat_ids = outputs['xzx']['z_hat_ids']
            z_ids, z_hat_ids = pad_label_label(z_ids[:, 1:], z_hat_ids, pad_token_id=self.pad_token_id)
            z_pad_mask = torch.logical_not(torch.eq(z_ids, self.pad_token_id))
            z_hat_ids = z_hat_ids * z_pad_mask

            x_pred_flag = torch.eq(x_hat_ids, x_ids).cpu().numpy()
            z_pred_flag = torch.eq(z_hat_ids, z_ids).cpu().numpy()
            
            correct_x_sentence_ids += torch.eq(x_hat_ids, x_ids).all(axis=-1).sum().cpu().numpy()
            correct_z_sentence_ids += torch.eq(z_hat_ids, z_ids).all(axis=-1).sum().cpu().numpy()
            # correct_x_sentence_ids = np.sum(np.all(x_pred_flag, axis=-1))
            # correct_z_sentence_ids = np.sum(np.all(z_pred_flag, axis=-1))
            
            correct_x_ids = correct_x_ids + np.sum(x_pred_flag) - torch.sum(torch.logical_not(x_pad_mask)).cpu().numpy()
            correct_z_ids = correct_z_ids + np.sum(z_pred_flag) - torch.sum(torch.logical_not(z_pad_mask)).cpu().numpy()

            total_x_ids = total_x_ids + torch.sum(x_pad_mask).cpu().numpy()
            total_z_ids = total_z_ids + torch.sum(z_pad_mask).cpu().numpy()

        return {'x_sentence_accuracy': correct_x_sentence_ids/len(dataloader.dataset), 
                'x_token_accuracy': correct_x_ids/total_x_ids,
                'z_sentence_accuracy': correct_z_sentence_ids/len(dataloader.dataset),
                'z_token_accuracy': correct_z_ids/total_z_ids}        

    def configure_optimizers(self):
        
        optimizer_grouped_parameters = [{"params": self.model_x_to_z.model.parameters()}, 
                                        {"params": self.model_z_to_x.model.parameters()},
                                        {"params": self.disc_x.parameters()},
                                        {"params": self.disc_z.parameters()}]
        
        if self.hparams.optimizer.get('_target_', False):
            model_x_to_z_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer, **optimizer_grouped_parameters[0])
            model_z_to_x_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer, **optimizer_grouped_parameters[1])
            disc_x_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer, **optimizer_grouped_parameters[2])
            disc_z_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer, **optimizer_grouped_parameters[3])

        else:
            model_x_to_z_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer.model_x_to_z, **optimizer_grouped_parameters[0])
            model_z_to_x_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer.model_z_to_x, **optimizer_grouped_parameters[1])
            disc_x_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer.disc_x, **optimizer_grouped_parameters[2])
            disc_z_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                self.hparams.optimizer.disc_z, **optimizer_grouped_parameters[3])
        
        
        # for pytorch scheduler objects, we should use utils.instantiate()
        if self.hparams.lr_scheduler["_target_"].startswith("torch.optim"):
            model_x_to_z_optimizer_scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler.model_x_to_z_scheduler, model_x_to_z_optimizer)
            model_z_to_x_optimizer_scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler.model_z_to_x_scheduler, model_z_to_x_optimizer)
            disc_x_optimizer_scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler.disc_x_scheduler, disc_x_optimizer)
            disc_z_optimizer_scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler.disc_z_scheduler, disc_z_optimizer)
            
        # for transformer function calls, we should use utils.call()
        elif self.hparams.lr_scheduler.formatter_scheduler["_target_"].startswith("transformers"):
            model_x_to_z_optimizer_scheduler = hydra.utils.call(
                self.hparams.lr_scheduler.model_x_to_z_scheduler, model_x_to_z_optimizer)
            model_z_to_x_optimizer_scheduler = hydra.utils.call(
                self.hparams.lr_scheduler.model_z_to_x_scheduler, model_z_to_x_optimizer)
            disc_x_optimizer_scheduler = hydra.utils.call(
                self.hparams.lr_scheduler.disc_x_scheduler, disc_x_optimizer)
            disc_z_optimizer_scheduler = hydra.utils.call(
                self.hparams.lr_scheduler.disc_z_scheduler, disc_z_optimizer)
            
        else:
            raise ValueError("The scheduler specified by scheduler._target_ is not implemented.")
        

        model_x_to_z_optimizer_dict = OmegaConf.to_container(self.hparams.lr_scheduler.model_x_to_z_scheduler_dict, resolve=True)
        model_x_to_z_optimizer_dict["scheduler"] = model_x_to_z_optimizer_scheduler
        model_z_to_x_optimizer_dict = OmegaConf.to_container(self.hparams.lr_scheduler.model_z_to_x_scheduler_dict, resolve=True)
        model_z_to_x_optimizer_dict["scheduler"] = model_z_to_x_optimizer_scheduler
        disc_x_optimizer_dict = OmegaConf.to_container(self.hparams.lr_scheduler.disc_x_scheduler_dict, resolve=True)
        disc_x_optimizer_dict["scheduler"] = disc_x_optimizer_scheduler
        disc_z_optimizer_dict = OmegaConf.to_container(self.hparams.lr_scheduler.disc_z_scheduler_dict, resolve=True)
        disc_z_optimizer_dict["scheduler"] = disc_z_optimizer_scheduler

        self.module_names = ['model_x_to_z', 'model_z_to_x', 'disc_x', 'disc_z']
        

        # formatter_scheduler_dict = OmegaConf.to_container(self.hparams.lr_scheduler.formatter_scheduler_dict, resolve=True)
        # formatter_scheduler_dict["scheduler"] = formatter_scheduler
        # reconstructor_scheduler_dict = OmegaConf.to_container(self.hparams.lr_scheduler.reconstructor_scheduler_dict, resolve=True)
        # reconstructor_scheduler_dict["scheduler"] = reconstructor_scheduler

        return [model_x_to_z_optimizer, model_z_to_x_optimizer, disc_x_optimizer, disc_z_optimizer], \
                [model_x_to_z_optimizer_dict, model_z_to_x_optimizer_dict, disc_x_optimizer_dict, disc_z_optimizer_dict]
   
    
    
    def _update_params(self, params, new_params):
        # for when you load pretrained model and want to update the params, 
        for k, v in new_params.items():
            if isinstance(v, collections.abc.Mapping):
                params[k] = self._update_params(params.get(k, {}), v)
            else:
                params[k] = v
        return params    
    
    # def _write_testing_output(self, step_summary):
    #     output_path = f"testing_output_{self.global_rank}.jsonl"

    #     if self.testing_output_parent_dir is not None:
    #         output_path = os.path.join(self.testing_output_parent_dir, output_path)

    #     with jsonlines.open(output_path, "a") as writer:
    #         writer.write_all(step_summary)

    
    def discretizer_dimensions(self):
        self.x_in_dim = self.hparams.blocks_model.hydra_configs.d_model
        self.z_in_dim = self.hparams.blocks_model.hydra_configs.d_model
        self.z_out_dim = self.hparams.blocks_model.hydra_configs.d_model
        self.x_out_dim = self.hparams.blocks_model.hydra_configs.d_model
        if self.hparams.model_params.use_tokenizer_vocab_len:     
            self.disc_x_vocab_size = self.collator.tokenizer_x.get_vocab_size()
            self.disc_z_vocab_size = self.collator.tokenizer_z.get_vocab_size()
        else:
            self.disc_x_vocab_size = self.hparams.model_params.disc_x_vocab_size
            self.disc_z_vocab_size = self.hparams.model_params.disc_z_vocab_size

        return( {'dimensions':{
                'encoder_embedding_dim': self.x_in_dim, 'decoder_embedding_dim': self.x_out_dim, 
                'vocab_size': self.disc_x_vocab_size, 'unembedding_dim': self.disc_x_vocab_size}},
                {'dimensions':{
                'encoder_embedding_dim': self.z_in_dim, 'decoder_embedding_dim': self.z_out_dim, 
                'vocab_size': self.disc_z_vocab_size, 'unembedding_dim': self.disc_z_vocab_size}})


    def dictionary_cosine_sim(self, alphabet='x'):
        if alphabet == 'x':
            kernel = self.disc_x.state_dict()['dictionary.weight'].cpu().numpy()
        elif alphabet == 'z':
            kernel = self.disc_z.state_dict()['dictionary.weight'].cpu().numpy()
        # cosine similarity
        inner_prods = kernel.dot(kernel.T)
        lengths = np.linalg.norm(kernel, axis=1)
        length_matrix = np.outer(lengths, lengths)
        kernel = np.round(inner_prods / length_matrix, decimals=2)
        return kernel


    def dictionary_inner_prod_sim(self, alphabet='x'):
        if alphabet == 'x':
            kernel = self.disc_x.state_dict()['dictionary.weight'].cpu().numpy()
        elif alphabet == 'z':
            kernel = self.disc_z.state_dict()['dictionary.weight'].cpu().numpy()
        inner_prods = kernel.dot(kernel.T)
        kernel = np.round(inner_prods, decimals=2)
        return kernel

    def format_matrix(self, matrix):
        formatted_rows = []
        for row in matrix:
            formatted_values = [f"{value: 8.2f}" for value in row]
            formatted_row = "[" + ", ".join(formatted_values) + "]"
            formatted_rows.append(formatted_row)
        return "\n\n".join(formatted_rows)