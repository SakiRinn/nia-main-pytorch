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
    input_words, output_words = datasets.read_test()
    dataset = ResDataset(input_words, output_words)
    data_loader = DataLoader(dataset,
                             batch_size=run_cfg['eval']['batch_size'],
                             num_workers=run_cfg['eval']['num_workers'],
                             shuffle=True,
                             pin_memory=True)

    # Model
    model_name = run_cfg['model']
    model_params = model_cfg[model_name]
    if model_name == 'Seq2Seq':
        del model_params['teacher_forcing_ratio']
    model = getter.get_model(run_cfg['model'],
                             dataset.input_vocab_len,
                             dataset.output_vocab_len,
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

            if model_name == 'Seq2Seq':
                out = model(src, trg, teacher_forcing_ratio=0.0)
                loss = loss_fn(out, trg).item()
                pred = model.pred2index(out)
            elif model_name == 'Transformer':
                src_key_padding_mask = mask.key_padding_mask(src).to(device)
                trg_key_padding_mask = mask.key_padding_mask(trg).to(device)
                attn_mask = mask.square_subsequent_mask(trg.size(1)).to(device)
                out = model(src, trg, src_key_padding_mask, trg_key_padding_mask, attn_mask)
                loss = loss_fn(out, trg, trg_key_padding_mask).item()
                pred = model.pred2index(out)
            else:
                out = model(src, trg)
                loss = loss_fn(out, trg).item()
                pred = model.pred2index(out)

            preds.append(pred)
            losses.append(loss)
            logger.info(f'[Step {step}] loss: {loss:.6f}')

        loss = torch.tensor(losses).mean().item()
        preds = torch.cat(preds, dim=0)
        logger.info(f'[RESULT] average loss: {loss:.6f}')

        inputs, outputs, results = [], [], []
        input_index_to_word, output_index_to_word = dataset.index_to_word()
        input_words, output_words = dataset.words()
        for input_text, output_text, result_text in zip(input_words, preds, output_words):
            entity = ' '.join([input_index_to_word[idx] for idx in input_text if idx > 0])
            pred_seq = ' '.join([output_index_to_word[idx] for idx in output_text if idx > 0])
            real_seq = ' '.join([output_index_to_word[idx] for idx in result_text if idx > 0])
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