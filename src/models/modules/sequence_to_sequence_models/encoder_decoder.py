from transformers import EncoderDecoderConfig, EncoderDecoderModel
from omegaconf import OmegaConf
import hydra

def EncoderDecoder(special_tokens_ids, **kwargs):        
    hparams = OmegaConf.create(kwargs)
            
    # if loading a pretrained model, but need to change some of the parameters
    
    config_encoder = hydra.utils.instantiate(hparams.config_encoder, _recursive_ = False)

    config_decoder = hydra.utils.instantiate(hparams.config_decoder, _recursive_ = False)

    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
    config.bos_token_id = special_tokens_ids['bos_token_id']
    config.eos_token_id = special_tokens_ids['eos_token_id']
    config.pad_token_id = special_tokens_ids['pad_token_id']

    config.output_hidden_states = True

    return(EncoderDecoderModel(config=config))

