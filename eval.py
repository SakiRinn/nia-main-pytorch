import datasets
from datasets import ResDataset
import utils.getter as getter
import utils.fileIO as fileIO
import utils.mask as mask
from modules.loss import MaskedLoss

import os
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader


def eval(ckpt_dir=''):
    # Config
    model_cfg = fileIO.load_config('./configs/model.yaml')
    run_cfg = fileIO.load_config('./configs/run.yaml')

    logger, eval_dir = fileIO.init_eval(ckpt_dir)
    device = run_cfg['device']

    # Dataset
    input_words, output_words = datasets.test_set()
    dataset = ResDataset(input_words, output_words)
    data_loader = DataLoader(dataset,
                             batch_size=run_cfg['eval']['batch_size'],
                             num_workers=run_cfg['eval']['num_workers'],
                             shuffle=True,
                             pin_memory=True)
    input_index_to_word, output_index_to_word = dataset.index_to_word()
    input_words, output_words = dataset.words()
    input_sos_index, output_sos_index = dataset.sos_index()
    input_eos_index, output_eos_index = dataset.eos_index()

    # Model
    model_name = run_cfg['model']
    model_params = model_cfg[model_name]
    if model_name == 'Seq2Seq':
        del model_params['teacher_forcing_ratio']
    model = getter.get_model(run_cfg['model'],
                             dataset.input_vocab_len,
                             dataset.output_vocab_len - 1,
                             **model_params).to(device)
    model.eval()

    # Load
    state = torch.load(ckpt_dir)
    model.load_state_dict(state)

    # Module
    loss_fn = getter.get_loss(run_cfg['eval']['loss'])
    if model_name == 'Transformer':
        loss_fn = MaskedLoss(loss_fn)

    # Main
    with torch.no_grad():
        preds = []
        losses = []

        for step, data in enumerate(tqdm(data_loader), start=1):
            src, trg = data[0].to(device), data[1].to(device)
            trg_in, trg_real = trg[:, 0:-1], trg[:, 1:]

            if model_name == 'Transformer':
                src_key_padding_mask = mask.key_padding_mask(src).to(device)
                trg_key_padding_mask = mask.key_padding_mask(trg_in).to(device)
                out, pred = model.predict(src, output_sos_index, output_eos_index, src_key_padding_mask, max_len=trg_real.size(1))
                loss = loss_fn(out, trg_real, trg_key_padding_mask).item()
            else:
                out, pred = model.predict(src, output_sos_index, output_eos_index, max_len=trg_real.size(1))
                loss = loss_fn(out, trg_real).item()

            preds.append(pred)
            losses.append(loss)
            logger.info(f'[Step {step}] loss: {loss:.6f}')

        loss = torch.tensor(losses).mean().item()
        preds = torch.cat(preds, dim=0).to(torch.long)
        logger.info(f'[RESULT] average loss: {loss:.6f}')

        inputs, outputs, results = [], [], []
        for input_text, output_text, result_text in zip(input_words, preds, output_words):
            entity = ' '.join([input_index_to_word[idx] for idx in input_text if idx not in [0, input_sos_index, input_eos_index]])
            pred_seq = ' '.join([output_index_to_word[idx] for idx in output_text if idx not in [0, output_sos_index, output_eos_index]])
            real_seq = ' '.join([output_index_to_word[idx] for idx in result_text if idx not in [0, output_sos_index, output_eos_index]])
            inputs.append(entity)
            outputs.append(pred_seq)
            results.append(real_seq)

        with open(os.path.join(eval_dir, f'[Pred] {ckpt_dir.split("/")[-1]}.txt'), 'w') as f:
            for input, output, result in zip(inputs, outputs, results):
                f.write('[ENTITY] ' + input + '\n')
                f.write('[ PRED ] ' + output + '\n')
                f.write('[ REAL ] ' + result + '\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str, default='', help='Path to a checkpoint file (.pth)')
    opt = parser.parse_args()

    eval(opt.checkpoint)