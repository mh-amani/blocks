import hydra
from pytorch_lightning import LightningModule
import collections.abc
import src.metrics
import os
import jsonlines
import json
import torch
from omegaconf import OmegaConf
from src.utils.metrics import pad_label_label
import numpy as np
from torchmetrics.wrappers import BootStrapper
import torchmetrics

class XZAutoencoder(LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        self.save_hyperparameters(ignore=["datamodule",])
                
        # if loading a pretrained model, but need to change some of the parameters
        if self.hparams.get('substitute_config'):
            self._update_params(self.hparams, self.hparams.substitute_config)
        
        self.special_tokens = self.hparams.special_tokens 
        self.pad_token_id = self.special_tokens.index('[pad]')
        self.eos_token_id = self.special_tokens.index('[eos]')
        self.bos_token_id = self.special_tokens.index('[bos]')
        self.unk_token_id = self.special_tokens.index('[unk]')
        self.special_tokens_ids = {'pad_token_id': self.pad_token_id, 'eos_token_id': self.eos_token_id, 
                                   'bos_token_id': self.bos_token_id}
        
        self.automatic_optimization = False
        self.decode_after_autoreg_step = self.hparams.model_params.decode_after_autoreg_step

        # the encoder and decoder
        self.model_x_to_z = hydra.utils.instantiate(self.hparams.modules.model_x_to_z, 
                                                    **self.hparams.modules.config_x_to_z,
                                                    special_tokens_ids=self.special_tokens_ids, _recursive_ = False)
        self.model_z_to_x = hydra.utils.instantiate(self.hparams.modules.model_z_to_x, 
                                                    **self.hparams.modules.config_z_to_x,
                                                    special_tokens_ids=self.special_tokens_ids, _recursive_ = False)
        
        # loss
        # old loss function in discretizer
        # return nn.CrossEntropyLoss(ignore_index=ignore_index)(preds.permute(0, 2, 1), label_ids)
        # smoothed_preds = (1 - self.label_smoothing_scale) * preds + self.label_smoothing_scale / self.vocab_size
        # self.loss = torch.nn.NLLLoss(ignore_index=self.pad_token_id)(torch.log(smoothed_preds).permute(0, 2, 1), label_ids)
        
        self.loss = torch.nn.NLLLoss(ignore_index=self.pad_token_id)
        self.loss_coeff = self.hparams.model_params.loss_coeff
        self.usexz = self.hparams.model_params['usexz']
        self.usez = self.hparams.model_params['usez']
        self.usex = self.hparams.model_params['usex']

        self.batch_size = self.hparams.dataset_parameters.batch_size

        self.max_x_length = self.hparams.model_params.max_x_length
        self.max_z_length = self.hparams.model_params.max_z_length

        self.acc_grad_batch = self.hparams.model_params.acc_grad_batch
        assert self.acc_grad_batch > 0, "acc_grad_batch must be greater than 0"

        
        # collate_fn
        train_dataset = kwargs['datamodule'].data_train
        self.pretokenized_flag = 0
        self.collator = hydra.utils.instantiate(self.hparams.collator, train_dataset,
                                                special_tokens=self.special_tokens, _recursive_ = False)
        
        # discretizers
        disc_conf_x, disc_conf_z = self.discretizer_dimensions()
        # discrete bottlenecks
        self.disc_x = hydra.utils.instantiate(self.hparams.modules.disc_x, disc_conf_x, _recursive_ = False)
        self.disc_z = hydra.utils.instantiate(self.hparams.modules.disc_z, disc_conf_z, _recursive_ = False)
        
                # Metrics
        # self.completeness = {'X': src.metrics.Completeness(), 'Z': src.metrics.Completeness()}
        # self.homogeneity = {'X': src.metrics.SentenceHomogeneity(), 'Z': src.metrics.SentenceHomogeneity()}
        # self.accuracy = {'X': src.metrics.Accuracy(self.pad_token_id).to(self.device), 
        #                  'Z': src.metrics.Accuracy(self.pad_token_id).to(self.device)}
        # self.accuracy_sentence = {'X': src.metrics.Accuracy(self.pad_token_id).to(self.device), 
        #                           'Z': src.metrics.Accuracy(self.pad_token_id).to(self.device)}
        # self.token_homogeneity = {'X': src.metrics.TokenHomogeneity(self.eos_token_id),
        #                            'Z': src.metrics.TokenHomogeneity(self.eos_token_id)}
        self.acc_mask= {'x': None, 'z': None}
        # numclasses = max(self.collator.tokenizer_x.get_vocab_size(), self.collator.tokenizer_z.get_vocab_size())
        numclasses = {'X': self.collator.tokenizer_x.get_vocab_size(), 'Z': self.collator.tokenizer_z.get_vocab_size()}
        self.accuracy = torch.nn.ModuleDict()
        self.accuracy_sentence = torch.nn.ModuleDict()
        self.manual_accuracy = {}
        self.manual_accuracy_sentence = {}
        for stage in ['val', 'test']:
            for type in ['teacherforced', 'autoreg', 'autoreg_hidden_layer']:
                for variable in ['X', 'Z']:
                    acc_name = f'{stage}/{type}/accuracy/{variable}'
                    sentence_acc_name = f'{stage}/{type}/sentence-accuracy/{variable}'
                    # self.accuracy[acc_name] = src.metrics.Accuracy(self.pad_token_id)
                    self.accuracy[acc_name] = torchmetrics.classification.MulticlassAccuracy(num_classes=numclasses[variable] ,ignore_index=self.pad_token_id)
                    self.accuracy_sentence[sentence_acc_name] = torchmetrics.classification.MulticlassExactMatch(num_classes=numclasses[variable] ,ignore_index=self.pad_token_id)
                    self.manual_accuracy[acc_name] = {'correct': 0, 'total': 0}
                    self.manual_accuracy_sentence[sentence_acc_name] = {'correct': 0, 'total': 0}

        self.wrong_x_predictions = []
        self.wrong_z_predictions = []
                    
    def setup(self, stage: str) -> None:
        pass
        # numclasses = max(self.collator.tokenizer_x.get_vocab_size(), self.collator.tokenizer_z.get_vocab_size())
        # self.accuracy = {}
        # self.accuracy_sentence = {}
        # for stage in ['train', 'val', 'test']:
        #     for type in ['teacherforced', 'autoreg', 'autoreg_hidden_layer']:
        #         for variable in ['X', 'Z']:
        #             acc_name = f'{stage}/{type}/accuracy/{variable}'
        #             sentence_acc_name = f'{stage}/{type}/sentence-accuracy/{variable}'
        #             self.accuracy[acc_name] = src.metrics.Accuracy(self.pad_token_id).to(self.device)
        #             self.accuracy_sentence[sentence_acc_name] = torchmetrics.classification.MulticlassExactMatch(num_classes=numclasses ,ignore_index=self.pad_token_id).to(self.device)


        # if self.hparams['collator']['tokenize_prior_training'] and self.pretokenized_flag==0:
        #     self.collator.pre_tokenize(self.trainer.datamodule.data_train)
        #     self.pretokenized_flag = 1

    def one_step_sequential_forward(self, model, discretizer, input_embeds, input_attention_mask, 
                                    output_embeds, output_attention_mask=None, past_key_values=None):
        
        if past_key_values is not None:
            output = model(inputs_embeds=input_embeds, attention_mask=input_attention_mask,
                           decoder_inputs_embeds=output_embeds, 
                           decoder_attention_mask=output_attention_mask,
                           past_key_values=past_key_values, output_hidden_states = True)
        else:
            output = model(inputs_embeds=input_embeds, attention_mask=input_attention_mask,
                           decoder_inputs_embeds=output_embeds, decoder_attention_mask=output_attention_mask,
                           output_hidden_states = True)
        
        output_embed = output['decoder_hidden_states'][-1]
        past_key_values = output['past_key_values']

        id, score, quantized_vector, quantization_loss = discretizer(output_embed)
        current_eos_flag = id == self.eos_token_id
        
        return(id, score, quantized_vector, quantization_loss, current_eos_flag, past_key_values)


    def sequential_forward(self, model, discretizer, input_embeds, input_attention_mask, output_embeds, max_length, output_attention_mask=None):
        
        eos_flag = torch.zeros(output_embeds.shape[0], 1, device=input_embeds.device)
        past_key_values = None
        quantization_loss = 0

        # first step to get the past_key_values
        id, score, quantized_vector, current_quantization_loss, current_eos_flag, past_key_values = \
            self.one_step_sequential_forward(model, discretizer, input_embeds, input_attention_mask,
                                                    output_embeds, output_attention_mask=output_attention_mask)
        
        quantization_loss += current_quantization_loss

        # if self.decode_after_autoreg_step:
        output_embed = discretizer.discrete_embedding_to_decoder(quantized_vector)

        # Added for doing average on quantization loss
        counter = 1
        
        while output_attention_mask.shape[1] < max_length and not torch.all(eos_flag):
            counter +=1
            current_id, current_score, current_quantized_vector, current_quantization_loss, current_eos_flag, past_key_values = \
            self.one_step_sequential_forward(model, discretizer, input_embeds, input_attention_mask,
                                                    output_embed, output_attention_mask=torch.logical_not(eos_flag),
                                                    past_key_values=past_key_values)
            
           
            id = torch.cat((id, current_id), dim=1)
            score = torch.cat((score, current_score), dim=1)
            quantized_vector = torch.cat((quantized_vector, current_quantized_vector), dim=1)
            quantization_loss += current_quantization_loss
            
            # if self.decode_after_autoreg_step:
            output_embed = discretizer.discrete_embedding_to_decoder(current_quantized_vector)

            eos_flag = torch.logical_or(eos_flag, current_eos_flag)
            output_attention_mask = torch.cat((output_attention_mask, torch.logical_not(eos_flag)), dim=1)
        
        return id, score, quantized_vector, quantization_loss/counter , output_attention_mask, eos_flag
        

    

    def forward_xzx(self, x_ids):
        # does there exist a simple huggging-facy way to do this? the following does not work
        # xz_out = self.model_x_to_z(inputs_embeds=x_embeds, decoder_inputs_embeds=x_embeds, output_hidden_states = True)['decoder_hidden_states'][-1]
        # xz_out = self.model_x_to_z.generate(inputs_embeds=x_embeds, max_length=x_embeds.shape[1], do_sample=False, output_hidden_states=True)

        x_embeds = self.disc_x.embed_enc_from_id(x_ids)
        x_attention_mask = torch.logical_not(torch.eq(x_ids, self.pad_token_id))

        output_attention_mask = torch.ones(x_embeds.shape[0], 1, device=x_embeds.device)

        z_ids = self.bos_token_id * torch.ones(x_embeds.shape[0], 1, device=x_embeds.device, dtype=torch.long)
        z_scores = torch.nn.functional.one_hot(z_ids, num_classes=self.disc_z.vocab_size).float()
        z_embeds = self.disc_z.embed_dec_from_id(z_ids)
        
        z_ids, z_scores, quantized_vector, z_quantization_loss, z_attention_mask, eos_flag = \
            self.sequential_forward(self.model_x_to_z, self.disc_z, x_embeds, x_attention_mask, z_embeds, self.max_z_length, output_attention_mask)
        
        x_embeds = self.disc_x.embed_dec_from_id(x_ids)
        z_embeds = self.disc_z.discrete_embedding_to_encoder(quantized_vector)

        x_hat_ids, x_hat_scores, _, x_quantization_loss = self.disc_x(self.model_z_to_x(inputs_embeds=z_embeds, attention_mask=z_attention_mask,
                                                decoder_inputs_embeds=x_embeds, decoder_attention_mask=x_attention_mask,
                                                output_hidden_states = True)['decoder_hidden_states'][-1])
        
        quantization_loss = z_quantization_loss + x_quantization_loss
        return {'x_hat_ids': x_hat_ids, 'x_hat_scores':x_hat_scores, 
                'z_hat_ids': z_ids, 'z_hat_scores': z_scores, 'quantization_loss': quantization_loss}     


    def forward_zxz(self, z_ids):
        z_embeds = self.disc_z.embed_enc_from_id(z_ids)
        z_attention_mask = torch.logical_not(torch.eq(z_ids, self.pad_token_id))

        output_attention_mask = torch.ones(z_embeds.shape[0], 1, device=z_embeds.device, dtype=torch.bool)

        x_ids = self.bos_token_id * torch.ones(z_embeds.shape[0], 1, device=z_embeds.device, dtype=torch.long)
        x_scores = torch.nn.functional.one_hot(x_ids, num_classes=self.disc_x.vocab_size).float()
        x_embeds = self.disc_x.embed_dec_from_id(x_ids)

        x_ids, x_scores, quantized_vector, x_quantization_loss, x_attention_mask, eos_flag = \
            self.sequential_forward(self.model_z_to_x, self.disc_x, z_embeds, z_attention_mask, x_embeds, self.max_x_length, output_attention_mask)

        z_embeds = self.disc_z.embed_dec_from_id(z_ids)
        x_embeds = self.disc_x.discrete_embedding_to_encoder(quantized_vector)

        z_hat_ids, z_hat_scores, _ , z_quantization_loss = self.disc_z(self.model_x_to_z(inputs_embeds=x_embeds, attention_mask=x_attention_mask,
                                                decoder_inputs_embeds=z_embeds, decoder_attention_mask=z_attention_mask,
                                                output_hidden_states = True)['decoder_hidden_states'][-1])

        quantization_loss = z_quantization_loss + x_quantization_loss 

        return {'x_hat_ids': x_ids, 'x_hat_scores': x_scores,
                'z_hat_ids': z_hat_ids, 'z_hat_scores': z_hat_scores, 'quantization_loss': quantization_loss}
         

    def forward_supervised_seperated(self, x_ids, z_ids):
        x_embeds_enc = self.disc_x.embed_enc_from_id(x_ids)        
        z_embeds_enc = self.disc_z.embed_enc_from_id(z_ids)
        x_embeds_dec = self.disc_x.embed_dec_from_id(x_ids)
        z_embeds_dec = self.disc_z.embed_dec_from_id(z_ids)
        x_attention_mask = torch.logical_not(torch.eq(x_ids, self.pad_token_id))
        z_attention_mask = torch.logical_not(torch.eq(z_ids, self.pad_token_id))

        out_z = self.model_x_to_z(inputs_embeds=x_embeds_enc, attention_mask=x_attention_mask,
                                    decoder_inputs_embeds=z_embeds_dec, decoder_attention_mask=z_attention_mask,
                                    output_hidden_states = True)['decoder_hidden_states'][-1]
        out_x = self.model_z_to_x(inputs_embeds=z_embeds_enc, attention_mask=z_attention_mask,
                                    decoder_inputs_embeds=x_embeds_dec, decoder_attention_mask=x_attention_mask,
                                    output_hidden_states = True)['decoder_hidden_states'][-1]
        
        x_hat_ids, x_hat_scores, _, x_quantization_loss = self.disc_x(out_x, supervision = True ,true_ids = x_ids)
        z_hat_ids, z_hat_scores, _, z_quantization_loss = self.disc_z(out_z, supervision = True, true_ids = z_ids)
        
        quantization_loss = x_quantization_loss + z_quantization_loss
        return {'x_hat_ids': x_hat_ids, 'x_hat_scores': x_hat_scores,
                'z_hat_ids': z_hat_ids, 'z_hat_scores': z_hat_scores, 'quantization_loss': quantization_loss}
    

    def forward(self, batch, stage='train'):
        data_type = batch['data_type']
        
        x_ids = batch['x_ids']
        z_ids = batch['z_ids']

        self.log(f"{stage}/x_data_available", float(data_type[0]), batch_size=self.batch_size)
        self.log(f"{stage}/z_data_available", float(data_type[1]), batch_size=self.batch_size)
        self.log('global_step', float(self.global_step), batch_size=self.batch_size)
        
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
        if (data_type[0] and data_type[1]) and (stage!='train' or self.usexz):
            output_supervised_seperated = self.forward_supervised_seperated(x_ids, z_ids)

            loss_x = self.loss(torch.log(output_supervised_seperated['x_hat_scores'][:, :-1, :]).permute(0, 2, 1), x_ids[:, 1:])
            loss_z = self.loss(torch.log(output_supervised_seperated['z_hat_scores'][:, :-1, :]).permute(0, 2, 1), z_ids[:, 1:])

            loss_supervised_seperated = self.loss_coeff['supervised_seperated_x'] * loss_x + self.loss_coeff['supervised_seperated_z'] * loss_z
            outputs['supervised_seperated'] = output_supervised_seperated
            losses['supervised_seperated'] = loss_supervised_seperated
            losses['supervised_seperated_x'] =  loss_x
            losses['supervised_seperated_z'] = loss_z
            losses['quantization_supervised_seperated'] = output_supervised_seperated['quantization_loss']

        # Unsupervized xzx pass
        if (data_type[0] and not data_type[1]) or (stage!='train') or (data_type[0] and data_type[1] and self.usex):
            output_xzx = self.forward_xzx(x_ids)
            loss_xzx = self.loss(torch.log(output_xzx['x_hat_scores'][:, :-1, :]).permute(0, 2, 1), x_ids[:, 1:])
            outputs['xzx'] = output_xzx
            losses['xzx'] = loss_xzx  
            losses['quantization_xzx'] = output_xzx['quantization_loss'] 
        
        
        # Unsupervized zxz pass
        if (data_type[1] and not data_type[0]) or (stage!='train') or (data_type[0] and data_type[1] and self.usez):
            output_zxz = self.forward_zxz(z_ids)
            loss_zxz = self.loss(torch.log(output_zxz['z_hat_scores'][:, :-1, :]).permute(0, 2, 1), z_ids[:, 1:])
            outputs['zxz'] = output_zxz
            losses['zxz'] = loss_zxz 
            losses['quantization_zxz'] = output_zxz['quantization_loss']
      
        loss = 0 
        for key in losses:
            if losses[key] is not None and self.loss_coeff.get(key) is not None:
                self.log(f'{stage}/loss/{key}', losses[key], batch_size=self.batch_size)
                loss += (self.loss_coeff[key]>0) * self.loss_coeff[key] * losses[key]  

        self.log(name=f'{stage}/loss', value=loss, batch_size=self.batch_size, prog_bar=True)  
        
        # for key in outputs:
        #     if outputs[key] is not None:
        #         for subkey in outputs[key]:
        #             self.log(f'{stage}/{key}/{subkey}', outputs[key][subkey])


        return loss, losses, outputs       
    
            
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.forward(batch)
        loss = loss / self.acc_grad_batch * 1.0
        
        optimizers = self.optimizers()
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        self.manual_backward(loss)
    
        if (batch_idx + 1) % self.acc_grad_batch == 0:
            for optimizer in optimizers:
                self.clip_gradients(optimizer, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                optimizer.step()
                optimizer.zero_grad()
            
            # self.disc_x.project_embedding_matrix()
            # self.disc_z.project_embedding_matrix()
        
        return loss


    def on_train_epoch_end(self):
        schedulers = self.lr_schedulers()
        for id, scheduler in enumerate(schedulers):
            # If the selected scheduler is a ReduceLROnPlateau scheduler.
            scheduler.step(self.trainer.callback_metrics[self.hparams.lr_scheduler.monitor])
            self.log(name=f'lr-scheduler/{self.module_names[id]}', value=scheduler._last_lr[0], batch_size=self.batch_size)

        # self.correct_predictions_mask()
        # print('-------on train epoch end------')
        # dict = self.correct_predictions_mask(self.trainer.datamodule.test_dataloader())
        # print('test_stats:', dict)
        # dict = self.correct_predictions_mask(self.trainer.datamodule.val_dataloader())
        # print('val_stats:', dict)
        # print('-----------------------')

    def compute_accuracy_measures(self, batch, batch_idx, stage):
        assert np.all(batch['data_type']), "compute_accuracy_measures: data_type must be supervised"
        
        _, _, outputs = self.forward(batch, stage=stage)
  
        accuracies = {}

        x_ids = batch['x_ids'].detach()
        z_ids = batch['z_ids'].detach()
        x_pad_mask = torch.logical_not(torch.eq(x_ids, self.pad_token_id))
        z_pad_mask = torch.logical_not(torch.eq(z_ids, self.pad_token_id))
        
        teacher_forced_available = outputs['supervised_seperated'] is not None
        autoreg_z_available = outputs['zxz'] is not None
        autoreg_x_available = outputs['xzx'] is not None

        if teacher_forced_available:
            x_hat_ids_teacherforced = outputs['supervised_seperated']['x_hat_ids'].detach()
            x_hat_ids_teacherforced[:, :-1] = x_hat_ids_teacherforced[:, :-1] * x_pad_mask[:, 1:]
            x_hat_ids_teacherforced[:, -1] = 0
            z_hat_ids_teacherforced = outputs['supervised_seperated']['z_hat_ids'].detach()
            z_hat_ids_teacherforced[:, :-1] = z_hat_ids_teacherforced[:, :-1] * z_pad_mask[:, 1:]
            z_hat_ids_teacherforced[:, -1] = 0
            x_ids_teacherforced, x_hat_ids_teacherforced = pad_label_label(x_ids, x_hat_ids_teacherforced, self.pad_token_id)
            z_ids_teacherforced, z_hat_ids_teacherforced = pad_label_label(z_ids, z_hat_ids_teacherforced, self.pad_token_id)
            
            self.accuracy_measures(x_ids_teacherforced, x_hat_ids_teacherforced, stage, 'X', 'teacherforced')
            self.accuracy_measures(z_ids_teacherforced, z_hat_ids_teacherforced, stage, 'Z', 'teacherforced')
        
        if autoreg_z_available:
            z_hat_ids_autoreg = outputs['zxz']['z_hat_ids'].detach()
            x_hat_ids_autoreg = outputs['zxz']['x_hat_ids'].detach()

            z_ids_autoreg, z_hat_ids_autoreg = pad_label_label(z_ids, z_hat_ids_autoreg, self.pad_token_id)
            x_ids_autoreg, x_hat_ids_autoreg = pad_label_label(x_ids, x_hat_ids_autoreg, self.pad_token_id)

            x_pad_mask = torch.logical_not(torch.eq(x_ids_autoreg, self.pad_token_id))
            z_pad_mask = torch.logical_not(torch.eq(z_ids_autoreg, self.pad_token_id))

            x_hat_ids_autoreg[:, :-1] = x_hat_ids_autoreg[:, :-1] * x_pad_mask[:, 1:]
            x_hat_ids_autoreg[:, -1] = 0
            z_hat_ids_autoreg[:, :-1] = z_hat_ids_autoreg[:, :-1] * z_pad_mask[:, 1:]
            z_hat_ids_autoreg[:, -1] = 0
            
            self.accuracy_measures(z_ids_autoreg, z_hat_ids_autoreg, stage, 'Z', 'autoreg')
            self.accuracy_measures(x_ids_autoreg, x_hat_ids_autoreg, stage, 'X', 'autoreg_hidden_layer')

         
        if autoreg_x_available:
            x_hat_ids_autoreg = outputs['xzx']['x_hat_ids'].detach() 
            z_hat_ids_autoreg = outputs['xzx']['z_hat_ids'].detach()
            
            x_ids_autoreg, x_hat_ids_autoreg = pad_label_label(x_ids, x_hat_ids_autoreg, self.pad_token_id)
            z_ids_autoreg, z_hat_ids_autoreg = pad_label_label(z_ids, z_hat_ids_autoreg, self.pad_token_id)

            x_pad_mask = torch.logical_not(torch.eq(x_ids_autoreg, self.pad_token_id))
            z_pad_mask = torch.logical_not(torch.eq(z_ids_autoreg, self.pad_token_id))

            x_hat_ids_autoreg[:, :-1] = x_hat_ids_autoreg[:, :-1] * x_pad_mask[:, 1:]
            x_hat_ids_autoreg[:, -1] = 0
            z_hat_ids_autoreg[:, :-1] = z_hat_ids_autoreg[:, :-1] * z_pad_mask[:, 1:]
            z_hat_ids_autoreg[:, -1] = 0

            self.accuracy_measures(x_ids_autoreg, x_hat_ids_autoreg, stage, 'X', 'autoreg')
            self.accuracy_measures(z_ids_autoreg, z_hat_ids_autoreg, stage, 'Z', 'autoreg_hidden_layer')

        return outputs

    def accuracy_measures(self, ids, hat_ids, stage, variable, type, log=True):
        
        # shifting to make the sequences aligned, removing bos
        acc_device = self.device
        ids = ids[:, 1:].to(acc_device)
        hat_ids = hat_ids[:, :-1].to(acc_device)

        acc_name = f'{stage}/{type}/accuracy/{variable}'
        sentence_acc_name = f'{stage}/{type}/sentence-accuracy/{variable}'
        
        pad_mask = torch.logical_not(torch.eq(ids, self.pad_token_id))
        hat_ids = hat_ids * pad_mask
        
        self.manual_accuracy[acc_name]['correct'] += torch.sum(torch.eq(hat_ids, ids)).cpu().numpy() - torch.sum(torch.logical_not(pad_mask)).cpu().numpy()
        self.manual_accuracy[acc_name]['total'] += torch.sum(pad_mask).cpu().numpy()
        
        self.manual_accuracy_sentence[sentence_acc_name]['correct'] += torch.sum(torch.eq(hat_ids, ids).all(axis=-1)).cpu().numpy()
        self.manual_accuracy_sentence[sentence_acc_name]['total'] += len(ids)
        

        # self.accuracy[acc_name].update(hat_ids.reshape(-1), ids.reshape(-1))
        # self.accuracy_sentence[sentence_acc_name].update(hat_ids, ids)
        # self.log(acc_name, self.accuracy[acc_name], batch_size=self.batch_size, )
        # self.log(sentence_acc_name, self.accuracy_sentence[sentence_acc_name], batch_size=self.batch_size)

        if sentence_acc_name.startswith('val/autoreg_hidden_layer/sentence-accuracy/X'):
            wrong_prediction = torch.where(torch.logical_not(torch.eq(hat_ids, ids).all(axis=-1)))
            self.wrong_x_predictions = self.wrong_x_predictions + wrong_prediction[0].cpu().numpy().tolist()
        
        if sentence_acc_name.startswith('val/autoreg_hidden_layer/sentence-accuracy/Z'):
            wrong_prediction = torch.where(torch.logical_not(torch.eq(hat_ids, ids).all(axis=-1)))
            self.wrong_z_predictions = self.wrong_z_predictions + wrong_prediction[0].cpu().numpy().tolist()

        # #Completeness test
        # value = self.completeness[variable](hat_ids, ids)
        # self.log(f'{stage}/{type}/completeness/{variable}', value, batch_size=self.batch_size)
            
        # #Homogeneity test
        # value = self.homogeneity[variable](hat_ids, ids)
        # self.log(f'{stage}/{type}/homogeneity/{variable}', value, batch_size=self.batch_size)

        # #Token homogeneity test
        # value = self.token_homogeneity[variable](hat_ids, ids)
        # self.log(f'{stage}/{type}/token_homogeneity/{variable}', value, batch_size=self.batch_size)

        # if self.hparams.get('write_testing_output', True):
        #     step_summary = {'stage': stage, 'type': type, 'x_ids': x_ids, 'x_hat_ids': x_hat_ids, 'z_ids': z_ids, 'z_hat_ids': z_hat_ids}
        #     self._write_testing_output(step_summary)
        # return {acc_name: acc, sentence_acc_name: acc_sentence}
    
    
    def validation_step(self, batch, batch_idx):
        self.compute_accuracy_measures(batch, batch_idx, stage='val')

    def on_validation_epoch_end(self):
        # self.correct_predictions_mask()
        for key in self.manual_accuracy:
            if self.manual_accuracy[key]['total'] > 0:   
                self.log('manual/' + key, self.manual_accuracy[key]['correct']/self.manual_accuracy[key]['total'], batch_size=self.batch_size)
                if key.startswith('val/autoreg_hidden_layer'):
                    print('manual/' + key, self.manual_accuracy[key]['correct']/self.manual_accuracy[key]['total'])
            self.manual_accuracy[key] = {'correct': 0, 'total': 0}
        for key in self.manual_accuracy_sentence:
            if self.manual_accuracy_sentence[key]['total'] > 0:
                self.log('manual/' + key, self.manual_accuracy_sentence[key]['correct']/self.manual_accuracy_sentence[key]['total'], batch_size=self.batch_size)
                if key.startswith('val/autoreg_hidden_layer'):
                    print('manual/' + key, self.manual_accuracy_sentence[key]['correct']/self.manual_accuracy_sentence[key]['total'])
            self.manual_accuracy_sentence[key] = {'correct': 0, 'total': 0}
        
        with open('wrong_x_predictions.json', 'w') as f:
            json.dump(self.wrong_x_predictions, f)
        with open('wrong_z_predictions.json', 'w') as f:
            json.dump(self.wrong_z_predictions, f)
        self.wrong_x_predictions = []
        self.wrong_z_predictions = []

        # for key in self.accuracy:
        #     self.accuracy[key].reset()
        # for key in self.accuracy_sentence:
        #     self.accuracy_sentence[key].reset()
            
        # print('-------on validation epoch end------')
        # dict = self.correct_predictions_mask(self.trainer.datamodule.test_dataloader())
        # print('test_stats:', dict)
        # dict = self.correct_predictions_mask(self.trainer.datamodule.val_dataloader())
        # print('val_stats:', dict)
        # print('-----------------------')

    def test_step(self, batch, batch_idx):
        
        outputs = self.compute_accuracy_measures(batch, batch_idx, stage='test')

        # if self.hparams['model_params'].get('num_bootstrap_tests', False):
        #     n = self.hparams['model_params']['num_bootstrap_tests']
        #     self.accuracy_bootstrapped = {'X': BootStrapper(src.metrics.Accuracy(self.pad_token_id).to(self.device),
        #                                         num_bootstraps=n).to(self.device), 
        #                      'Z': BootStrapper(src.metrics.Accuracy(self.pad_token_id).to(self.device),
        #                                         num_bootstraps=n).to(self.device)}
        #     self.accuracy_sentence_bootsrapped = {'X': BootStrapper(torchmetrics.classification.MulticlassExactMatch(num_classes=self.disc_x_vocab_size ,ignore_index=self.pad_token_id).to(self.device),
        #                                                  num_bootstraps=n).to(self.device),
        #                                 'Z': BootStrapper(torchmetrics.classification.MulticlassExactMatch(num_classes=self.disc_z_vocab_size ,ignore_index=self.pad_token_id).to(self.device),
        #                                                    num_bootstraps=n).to(self.device)}
              
        #     x_ids = batch['x_ids'].detach()
        #     z_ids = batch['z_ids'].detach()
        #     x_pad_mask = torch.logical_not(torch.eq(x_ids, self.pad_token_id))
        #     z_pad_mask = torch.logical_not(torch.eq(z_ids, self.pad_token_id))
            

        #     x_hat_ids_autoreg = outputs['zxz']['x_hat_ids'].detach()
        #     x_ids_autoreg, x_hat_ids_autoreg = pad_label_label(x_ids, x_hat_ids_autoreg, self.pad_token_id)
        #     x_pad_mask = torch.logical_not(torch.eq(x_ids_autoreg, self.pad_token_id))
        #     x_hat_ids_autoreg[:, :-1] = x_hat_ids_autoreg[:, :-1] * x_pad_mask[:, 1:]
        #     x_hat_ids_autoreg[:, -1] = 0
        #     x_acc= self.accuracy_bootstrapped['X'](x_hat_ids_autoreg.reshape(-1), x_ids_autoreg.reshape(-1))
        #     x_acc_sentence = self.accuracy_sentence_bootsrapped['X'](x_hat_ids_autoreg, x_ids_autoreg)

        #     self.log('test/accuracy-mean/X', x_acc['mean'], batch_size=self.batch_size)
        #     self.log('test/accuracy-std/X', x_acc['std'], batch_size=self.batch_size)
        #     self.log('test/sentence-accuracy-mean/X', x_acc_sentence['mean'], batch_size=self.batch_size)
        #     self.log('test/sentence-accuracy-std/X', x_acc_sentence['std'], batch_size=self.batch_size)
            
        #     z_hat_ids_autoreg = outputs['xzx']['z_hat_ids'].detach()
        #     z_ids_autoreg, z_hat_ids_autoreg = pad_label_label(z_ids, z_hat_ids_autoreg, self.pad_token_id)
        #     z_pad_mask = torch.logical_not(torch.eq(z_ids_autoreg, self.pad_token_id))
        #     z_hat_ids_autoreg[:, :-1] = z_hat_ids_autoreg[:, :-1] * z_pad_mask[:, 1:]
        #     z_hat_ids_autoreg[:, -1] = 0
        #     z_acc = self.accuracy_bootstrapped['Z'](z_hat_ids_autoreg.reshape(-1), z_ids_autoreg.reshape(-1))
        #     z_acc_sentence = self.accuracy_sentence_bootsrapped['Z'](z_hat_ids_autoreg, z_ids_autoreg)

        #     self.log('test/accuracy-mean/Z', z_acc['mean'], batch_size=self.batch_size)
        #     self.log('test/accuracy-std/Z', z_acc['std'], batch_size=self.batch_size)
        #     self.log('test/sentence-accuracy-mean/Z', z_acc_sentence['mean'], batch_size=self.batch_size)
        #     self.log('test/sentence-accuracy-std/Z', z_acc_sentence['std'], batch_size=self.batch_size)

    
    def on_test_epoch_end(self):
        # self.correct_predictions_mask()
        for key in self.manual_accuracy:
            if self.manual_accuracy[key]['total'] > 0:   
                self.log('manual/' + key, self.manual_accuracy[key]['correct']/self.manual_accuracy[key]['total'], batch_size=self.batch_size)
            self.manual_accuracy[key] = {'correct': 0, 'total': 0}
        for key in self.manual_accuracy_sentence:
            if self.manual_accuracy_sentence[key]['total'] > 0:
                self.log('manual/' + key, self.manual_accuracy_sentence[key]['correct']/self.manual_accuracy_sentence[key]['total'], batch_size=self.batch_size)
            self.manual_accuracy_sentence[key] = {'correct': 0, 'total': 0}
        # dict = self.correct_predictions_mask(self.trainer.datamodule.test_dataloader())
        # print('test_stats:', dict)
        # dict = self.correct_predictions_mask(self.trainer.datamodule.val_dataloader())
        # print('val_stats:', dict)

    

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
