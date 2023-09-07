import hydra
from pytorch_lightning import LightningModule
import collections.abc
import src.metrics
import os
import jsonlines
import torch
from omegaconf import OmegaConf
from src.utils.metrics import pad_label_label, pad_logit_label
import numpy as np
from src.utils.general import data_type_str


class XZAutoencoder(LightningModule):
    def __init__(self, datamodule=None, **kwargs) -> None:
        super().__init__()
        
        self.save_hyperparameters()

        self.datamodule = datamodule
                
        # if loading a pretrained model, but need to change some of the parameters
        if self.hparams.get('checkpoint_path') and self.hparams.get('substitute_config'):
            self._update_params(self.hparams, self.hparams.substitute_config)
        
        self.special_tokens = self.hparams.special_tokens 
        self.pad_token_id = self.special_tokens.index('[pad]')
        self.eos_token_id = self.special_tokens.index('[eos]')
        self.bos_token_id = self.special_tokens.index('[bos]')
        self.unk_token_id = self.special_tokens.index('[unk]')
        self.special_tokens_ids = {'pad_token_id': self.pad_token_id, 'eos_token_id': self.eos_token_id, 
                                   'bos_token_id': self.bos_token_id}
        self.automatic_optimization = False
        self.tokenize_after_generation = self.hparams.model_params.tokenize_after_generation
        # the encoder and decoder
        self.model_x_to_z = hydra.utils.instantiate(self.hparams.modules.model_x_to_z,
                                                     special_tokens_ids=self.special_tokens_ids, _recursive_ = False)
        self.model_z_to_x = hydra.utils.instantiate(self.hparams.modules.model_z_to_x, 
                                                     special_tokens_ids=self.special_tokens_ids, _recursive_ = False)
        
        # loss function coefficients
        self.reconstruction_loss_coeff_x = self.hparams.model_params.reconstruction_loss_coeff_x
        self.reconstruction_loss_coeff_z = self.hparams.model_params.reconstruction_loss_coeff_z

        # self.model.pad_token_id = -100

        # Metrics
        self.homogeneity_z = src.metrics.SentenceHomogeneity()
        self.token_homogeneity_z = src.metrics.TokenHomogeneity(self.eos_token_id)
        self.completeness_z = src.metrics.Completeness()
        self.accuracy_z = src.metrics.Accuracy(self.pad_token_id)
        self.accuracy_z_sentence = src.metrics.Accuracy(self.pad_token_id)
        self.homogeneity_x = src.metrics.SentenceHomogeneity()
        self.token_homogeneity_x = src.metrics.TokenHomogeneity(self.eos_token_id)
        self.completeness_x = src.metrics.Completeness()
        self.accuracy_x = src.metrics.Accuracy(self.pad_token_id)
        self.accuracy_x_sentence = src.metrics.Accuracy(self.pad_token_id)

        self.max_x_length = self.hparams.model_params.max_x_length
        self.max_z_length = self.hparams.model_params.max_z_length


    def setup(self, stage: str) -> None:
        disc_conf_x, disc_conf_z = self.discretizer_dimensions()
        # discrete bottlenecks
        self.disc_x = hydra.utils.instantiate(self.hparams.modules.disc_x, disc_conf_x, self.pad_token_id, _recursive_ = False)
        self.disc_z = hydra.utils.instantiate(self.hparams.modules.disc_z, disc_conf_z, self.pad_token_id, _recursive_ = False)


    def one_step_sequential_forward(self, model, discretizer, input_embeds, output_embeds, eos_flag, past_key_values=None):
        if past_key_values is not None:
            output = model(inputs_embeds=input_embeds, decoder_inputs_embeds=output_embeds[..., -1:, :], past_key_values=past_key_values,
                                     output_hidden_states = True)
        else:
            output = model(inputs_embeds=input_embeds, decoder_inputs_embeds=output_embeds ,output_hidden_states = True)
        
        output_embed = output['decoder_hidden_states'][-1]
        past_key_values = output['past_key_values']

        current_eos_flag = discretizer.decode(discretizer.discretize(output_embed)) == self.eos_token_id
        
        return(output_embed, current_eos_flag, past_key_values)


    def sequential_forward(self, model, discretizer, input_embeds, bos_embed):
        output_embeds = torch.zeros(input_embeds.shape[0], 1, input_embeds.shape[2], device=input_embeds.device)
        output_embeds[:, 0, :] = bos_embed
        eos_flag = torch.zeros(input_embeds.shape[0], 1, device=input_embeds.device)
        past_key_values = None
        if self.tokenize_after_generation:
            while output_embeds.shape[1] < self.max_x_length and not torch.all(eos_flag):
                output_embed, current_eos_flag, past_key_values = self.one_step_sequential_forward(model, discretizer, 
                                                                                        input_embeds, output_embeds, 
                                                                                        eos_flag, past_key_values)
                
                output_embeds = torch.cat((output_embeds, output_embed), dim=1)
                eos_flag = torch.logical_or(eos_flag, current_eos_flag)

            output_embeds = discretizer.discretize(output_embeds)

        else:
            while output_embeds.shape[1] < self.max_x_length and not torch.all(eos_flag):
                output_embed, current_eos_flag, past_key_values = self.one_step_sequential_forward(model, discretizer,
                                                                                        input_embeds, output_embeds,
                                                                                        eos_flag)
                output_embed = discretizer.discretize(output_embed)
                output_embeds = torch.cat((output_embeds, output_embed), dim=1)
                eos_flag = torch.logical_or(eos_flag, current_eos_flag)


        return {'decoder_hidden_states': output_embeds, 'eos_flag': eos_flag}

    

    def forward_xzx(self, x_ids):
        # does there exist a simple huggging-facy way to do this? the following does not work
        # xz_out = self.model_x_to_z(inputs_embeds=x_embeds, decoder_inputs_embeds=x_embeds, output_hidden_states = True)['decoder_hidden_states'][-1]
        # xz_out = self.model_x_to_z.generate(inputs_embeds=x_embeds, max_length=x_embeds.shape[1], do_sample=False, output_hidden_states=True)

        x_embeds = self.disc_x.embed_from_id(x_ids)
        xz_out= self.sequential_forward(self.model_x_to_z, self.disc_z, x_embeds, 
                                                      self.disc_x.embed_from_id(torch.tensor([self.bos_token_id], 
                                                                                             device=x_embeds.device)))['decoder_hidden_states']
        
        z_hat = self.disc_z.decode(xz_out)
        
        zx_in = self.disc_z.embed_from_discrete_representation(xz_out)

        x_hat = self.disc_x.discretize(self.model_z_to_x(inputs_embeds=zx_in, 
                                                         decoder_inputs_embeds=x_embeds,
                                                         output_hidden_states = True)['decoder_hidden_states'][-1])

        return {'x_embeds': x_embeds, 'xz_out': xz_out, 'z_hat': z_hat, 'x_hat': x_hat}


    def forward_zxz(self, z_ids):
        z_embeds = self.disc_z.embed_from_id(z_ids)
        zx_out= self.sequential_forward(self.model_z_to_x, self.disc_x, z_embeds, 
                                                      self.disc_z.embed_from_id(torch.tensor([self.bos_token_id], 
                                                                                             device=z_embeds.device)))['decoder_hidden_states']
        
        x_hat = self.disc_x.decode(zx_out)
        
        xz_in = self.disc_x.embed_from_discrete_representation(zx_out)

        z_hat = self.disc_z.discretize(self.model_x_to_z(inputs_embeds=xz_in, 
                                                         decoder_inputs_embeds=z_embeds,
                                                         output_hidden_states = True)['decoder_hidden_states'][-1])

        return {'x_embeds': z_embeds, 'zx_out': zx_out, 'x_hat': x_hat, 'z_hat': z_hat}
    

    def forward_supervised_seperated(self, x_ids, z_ids):
        x_embeds = self.disc_x.embed_from_id(x_ids)        
        z_embeds = self.disc_z.embed_from_id(z_ids)

        z_hat = self.disc_z.discretize(self.model_x_to_z(inputs_embeds=x_embeds, 
                                                         decoder_inputs_embeds=z_embeds,
                                                         output_hidden_states = True)['decoder_hidden_states'][-1])
        x_hat = self.disc_x.discretize(self.model_z_to_x(inputs_embeds=z_embeds,
                                                            decoder_inputs_embeds=x_embeds,
                                                            output_hidden_states = True)['decoder_hidden_states'][-1])

        return {'x_embeds': x_embeds, 'x_hat': x_hat, 'z_embeds': z_embeds, 'z_hat': z_hat}
    

    def forward(self, batch, stage='train'):
        data_type = batch['data_type']
        
        x_ids = batch['x_ids']
        z_ids = batch['z_ids']

        # logging the supervision type, 10 for only x, 01 for only z, 11 for both :/ 
        self.log_dict({f"{stage}/supervision": data_type[0]*10 + data_type[1] ,'global_step': self.global_step})
        
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

        if data_type[0]:
            output_xzx = self.forward_xzx(x_ids)
            loss_xzx = self.disc_x.loss(output_xzx['x_hat'], x_ids)
            outputs['xzx'] = output_xzx
            losses['xzx'] = loss_xzx   
        if data_type[1]:
            output_zxz = self.forward_zxz(z_ids)
            loss_zxz = self.disc_z.loss(output_zxz['z_hat'], z_ids)
            outputs['zxz'] = output_zxz
            losses['zxz'] = loss_zxz 
        if data_type[0] and data_type[1]:
            output_supervised_seperated = self.forward_supervised_seperated(x_ids, z_ids)
            loss_x = self.disc_x.loss(output_supervised_seperated['x_hat'], x_ids)
            loss_z = self.disc_z.loss(output_supervised_seperated['z_hat'], z_ids)
            loss_supervised_seperated = self.reconstruction_loss_coeff_x * loss_x + self.reconstruction_loss_coeff_z * loss_z
            outputs['supervised_seperated'] = output_supervised_seperated
            losses['supervised_seperated'] = loss_supervised_seperated
            losses['supervised_seperated_x'] = loss_x
            losses['supervised_seperated_z'] = loss_z
            
        for key in losses:
            self.log(f'{stage}/{key}', losses[key])
        
        # for key in outputs:
        #     if outputs[key] is not None:
        #         for subkey in outputs[key]:
        #             self.log(f'{stage}/{key}/{subkey}', outputs[key][subkey])

            

        # self.log(f'{stage}/loss_xzx', loss_xzx)
        # self.log(f'{stage}/loss_zxz', loss_zxz)
        # self.log(f'{stage}/loss_supervised_seperated_x', loss_x)
        # self.log(f'{stage}/loss_supervised_seperated_z', loss_z)
        # self.log(f'{stage}/loss_supervised_seperated', loss_supervised_seperated)

        return outputs, losses       
    
            
    def training_step(self, batch, batch_idx):
        outputs, losses = self.forward(batch)
        
        loss = 0
        for key in losses:
            if losses[key] is not None:
                loss += losses[key]

        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        self.manual_backward(loss)
    
        # clip gradients
        for optimizer in optimizers:
            self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optimizer.step()
        
        return loss
    
    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        for scheduler in schedulers:
            # If the selected scheduler is a ReduceLROnPlateau scheduler.
            scheduler.step(self.trainer.callback_metrics[self.hparams.lr_scheduler.monitor])
            self.log_dict({"reconstructor_LR":scheduler._last_lr[0],'global_step': self.global_step})


    def compute_accuracy_measures(self, batch, batch_idx, stage):
        assert np.all(batch['data_type']), "compute_accuracy_measures: data_type must be supervised"
        
        outputs, losses = self.forward(batch, stage='val')

        x_ids = batch['x_ids'].detach()
        z_ids = batch['z_ids'].detach()
        
        x_hat_ids_teacherforced = self.disc_x.decode(outputs['supervised_seperated']['x_hat'].detach())
        z_hat_ids_teacherforced = self.disc_z.decode(outputs['supervised_seperated']['z_hat'].detach())
        x_hat_ids_autoreg = outputs['zxz']['x_hat'].detach()
        z_hat_ids_autoreg = outputs['xzx']['z_hat'].detach()

        #Pad labels to similar length
        x_ids_autoreg, x_hat_ids_autoreg = pad_label_label(x_ids, x_hat_ids_autoreg, self.pad_token_id)
        z_ids_autoreg, z_hat_ids_autoreg = pad_label_label(z_ids, z_hat_ids_autoreg, self.pad_token_id)
        x_ids_teacherforced, x_hat_ids_teacherforced = pad_label_label(x_ids, x_hat_ids_teacherforced, self.pad_token_id)
        z_ids_teacherforced, z_hat_ids_teacherforced = pad_label_label(z_ids, z_hat_ids_teacherforced, self.pad_token_id)
        
        self.accuracy_measures(x_hat_ids_teacherforced, z_hat_ids_teacherforced, 
                               x_ids_teacherforced, z_ids_teacherforced, stage, 'teacherforced')
        self.accuracy_measures(x_hat_ids_autoreg, z_hat_ids_autoreg, x_ids_autoreg, z_ids_autoreg, stage, 'autoreg')
        
        return outputs


    def accuracy_measures(self, x_hat_ids, z_hat_ids, x_ids, z_ids, stage, type):
        #Completeness test
        value = self.completeness_x(x_ids, x_hat_ids)
        self.log(f'{stage}_{type}_completeness_x', value)
        value = self.completeness_z(z_ids, z_hat_ids)
        self.log(f'{stage}_{type}_completeness_z', value)
        
        #Homogeneity test
        value = self.homogeneity_x(x_ids, x_hat_ids)
        self.log(f'{stage}_{type}_homogeneity_x', value)
        value = self.homogeneity_z(z_ids, z_hat_ids)
        self.log(f'{stage}_{type}_homogeneity_z', value)

        #Accuracy test
        value = self.accuracy_x(x_ids, x_hat_ids)
        self.log(f'{stage}_{type}_accuracy_x', value)
        value = self.accuracy_z(z_ids, z_hat_ids)
        self.log(f'{stage}_{type}_accuracy_z', value)

        #Accuracy sentence test
        value = self.accuracy_x_sentence(x_ids, x_hat_ids)
        self.log(f'{stage}_{type}_accuracy_x_sentence', value)
        value = self.accuracy_z_sentence(z_ids, z_hat_ids)
        self.log(f'{stage}_{type}_accuracy_z_sentence', value)

        #Token homogeneity test
        value = self.token_homogeneity_x(x_ids, x_hat_ids)
        self.log(f'{stage}_{type}_token_homogeneity_x', value)
        value = self.token_homogeneity_z(z_ids, z_hat_ids)
        self.log(f'{stage}_{type}_token_homogeneity_z', value)

        # if self.hparams.get('write_testing_output', True):
        #     step_summary = {'stage': stage, 'type': type, 'x_ids': x_ids, 'x_hat_ids': x_hat_ids, 'z_ids': z_ids, 'z_hat_ids': z_hat_ids}
        #     self._write_testing_output(step_summary)

        
    def validation_step(self, batch, batch_idx):
        output = self.compute_accuracy_measures(batch, batch_idx, 'val')
        return output
    

    def test_step(self, batch, batch_idx):
        output = self.compute_accuracy_measures(batch, stage='test')
        return output
    
    
    def configure_optimizers(self):
        
        optimizer_grouped_parameters = [{"params": self.model_x_to_z.parameters()}, 
                                        {"params": self.model_z_to_x.parameters()},
                                        {"params": self.disc_x.parameters()},
                                        {"params": self.disc_z.parameters()}]
        
        model_x_to_z_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, **optimizer_grouped_parameters[0])
        model_z_to_x_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, **optimizer_grouped_parameters[1])
        disc_x_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, **optimizer_grouped_parameters[2])
        disc_z_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, **optimizer_grouped_parameters[3])
        
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
        

        # formatter_scheduler_dict = OmegaConf.to_container(self.hparams.lr_scheduler.formatter_scheduler_dict, resolve=True)
        # formatter_scheduler_dict["scheduler"] = formatter_scheduler
        # reconstructor_scheduler_dict = OmegaConf.to_container(self.hparams.lr_scheduler.reconstructor_scheduler_dict, resolve=True)
        # reconstructor_scheduler_dict["scheduler"] = reconstructor_scheduler

        return [model_x_to_z_optimizer, model_z_to_x_optimizer, disc_x_optimizer, disc_z_optimizer], \
                [model_x_to_z_optimizer_dict, model_z_to_x_optimizer_dict, disc_x_optimizer_dict, disc_z_optimizer_dict]
    # [model_x_to_z_optimizer_scheduler, model_z_to_x_optimizer_scheduler, disc_x_optimizer_scheduler, disc_z_optimizer_scheduler]
        #
        # return [formatter_optimizer, reconstructor_optimizer], [formatter_scheduler_dict, reconstructor_scheduler_dict]
    
    
    def _update_params(self, params, new_params):
        # for when you load pretrained model and want to update the params, 
        for k, v in new_params.items():
            if isinstance(v, collections.abc.Mapping):
                params[k] = self.update_params(params.get(k, {}), v)
            else:
                params[k] = v
        return params
    
    
    def _write_testing_output(self, step_summary):
        output_path = f"testing_output_{self.global_rank}.jsonl"

        if self.testing_output_parent_dir is not None:
            output_path = os.path.join(self.testing_output_parent_dir, output_path)

        with jsonlines.open(output_path, "a") as writer:
            writer.write_all(step_summary)

    
    def discretizer_dimensions(self):
        self.x_in_dim = self.model_x_to_z.config.encoder.hidden_size
        self.z_in_dim = self.model_z_to_x.config.encoder.hidden_size
        self.z_out_dim = self.model_x_to_z.config.decoder.hidden_size
        self.x_out_dim = self.model_z_to_x.config.decoder.hidden_size
        if self.hparams.model_params.use_tokenizer_vocab_len:     
            self.disc_x_vocab_size = self.collator.tokenizer_x.get_vocab_size()
            self.disc_z_vocab_size = self.collator.tokenizer_z.get_vocab_size()
        else:
            self.disc_x_vocab_size = self.hparams.model_params.disc_x_vocab_size
            self.disc_z_vocab_size = self.hparams.model_params.disc_z_vocab_size
        return( {'input_dim': self.x_in_dim, 'output_dim': self.x_out_dim, 'vocab_size': self.disc_x_vocab_size}, 
                {'input_dim': self.z_in_dim, 'output_dim': self.z_out_dim, 'vocab_size': self.disc_z_vocab_size})


    def setup_collate_fn(self, datamodule):
        train_dataset = datamodule.data_train
        self.collator = hydra.utils.instantiate(self.hparams.collator, train_dataset, _recursive_ = False)
