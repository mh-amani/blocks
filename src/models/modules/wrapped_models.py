from transformers import T5Tokenizer, T5Model
from blocks.modules.auto_reg_wrapper import AutoRegWrapper
from blocks.unwrapped_models.enc_dec_unwrapper import Unwrappedbart
from blocks.modules.blocks_connector import BlocksConnector
import hydra

def UntrainedBart(disc_dims_x, disc_dims_z, hydra_configs):
    # initialize the vector models
    pad_token_id = hydra_configs.special_tokens.index('[pad]')        
    eos_token_id = hydra_configs.special_tokens.index('[eos]')
    bos_token_id = hydra_configs.special_tokens.index('[bos]')
    unk_token_id = hydra_configs.special_tokens.index('[unk]')
    tokenizer_config = {'control_token_ids': {
                                'input_pad_token_id': pad_token_id,
                                'output_eos_token_id': eos_token_id, 
                                'output_pad_token_id': pad_token_id,
                                'output_unknown_token_id': unk_token_id,},
            '               output_prepending_ids': bos_token_id
                            }


    vector_model_x_z, _, _, _ = Unwrappedbart(hydra.utils.instantiate(hydra_configs.config_x_to_z))
    vector_model_z_x, _, _, _ = Unwrappedbart(hydra.utils.instantiate(hydra_configs.config_z_to_x))  
    # initializing the discretizers
    disc_x = hydra.utils.instantiate(hydra_configs.disc_x, {**hydra_configs.disc_x_config, **disc_dims_x})
    disc_z = hydra.utils.instantiate(hydra_configs.disc_z, {**hydra_configs.disc_z_config, **disc_dims_z})

    
    model_x_to_z = AutoRegWrapper(vector_model_x_z, disc_x, disc_z,
                                        config={**hydra_configs.autoreg_wrapper_config,
                                                    **tokenizer_config, 'device': 'cpu', 
                                                    'output_prepending_ids': bos_token_id})
    model_z_to_x = AutoRegWrapper(vector_model_z_x, disc_z, disc_x,
                                        config={**hydra_configs.autoreg_wrapper_config,
                                                    **tokenizer_config, 'device': 'cpu', 
                                                    'output_prepending_ids': bos_token_id})

    def transform_xy_outputs_to_y_inputs(xy_outputs):
        # since bart output has a eos <\s> token prepended in its output, we remove it for feeding to the next model
        return {'output_attention_mask': xy_outputs['output_attention_mask'],
                'quantized_vector_encoder': xy_outputs['quantized_vector_encoder']}

    model_x_to_z_to_x = BlocksConnector(model_x_to_z, model_z_to_x, config=None)
    model_x_to_z_to_x.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs
    model_z_to_x_to_z = BlocksConnector(model_z_to_x, model_x_to_z, config=None)
    model_z_to_x_to_z.transform_xy_outputs_to_y_inputs = transform_xy_outputs_to_y_inputs

    return(disc_x, disc_z, model_x_to_z, model_z_to_x, model_x_to_z_to_x, model_z_to_x_to_z)