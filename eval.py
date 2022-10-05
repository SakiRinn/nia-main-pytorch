import datasets
from datasets import ResDataset
import utils.getter as getter
import utils.fileIO as fileIO
import utils.mask as mask
from modules.loss import MaskedLoss

from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader


def eval(ckpt_dir=''):
    # Config
    model_cfg = fileIO.load_config('./configs/model.yaml')
    run_cfg = fileIO.load_config('./configs/run.yaml')

    logger = fileIO.init_eval(ckpt_dir)
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
        teacher_forcing_ratio = model_params['teacher_forcing_ratio']
        del model_params['teacher_forcing_ratio']
    model = getter.get_model(run_cfg['model'],
                             dataset.input_vocab_len,
                             dataset.output_vocab_len,
                             **model_params).to(device)
    model.eval()

    # Load
    model = torch.load(ckpt_dir)
    model.eval()

    # Module
    # loss_fn = getter.get_loss(run_cfg['eval']['loss'])
    # if model_name == 'Transformer':
    #     loss_fn = MaskedLoss(loss_fn)

    # Main
    with torch.no_grad():
        for step, data in enumerate(tqdm(data_loader), start=1):
            src, trg = data[0].to(device), data[1].to(device)

            preds = []
            losses = []
            if model_name == 'Transformer':
                src_key_padding_mask = mask.key_padding_mask(src).to(device)
                trg_key_padding_mask = mask.key_padding_mask(trg).to(device)
                attn_mask = mask.square_subsequent_mask(trg.size(1)).to(device)
                pred = model.predict(src, trg, src_key_padding_mask, trg_key_padding_mask, attn_mask)
                # loss = loss_fn(pred, trg, trg_key_padding_mask).item()
            else:
                pred = model.predict(src, trg)
                # loss = loss_fn(pred, trg).item()
            # logger.info(f'[Step {step}] loss: {loss}')
            # losses.append(loss)

        # loss = torch.tensor(losses).mean().item()
        # logger.info(f'[RESULT] average loss: {loss}')

        inputs, outputs = [], []
        input_index_to_word, output_index_to_word = dataset.index_to_word()
        for input_sequence, pred in zip(input_words, preds):
            entities = ' '.join([input_index_to_word[idx] for idx in input_sequence if idx > 0])
            sequence = ' '.join([output_index_to_word[idx] for idx in pred if idx > 0])
            logger.info(entities)
            logger.info(sequence)
            inputs.append(entities)
            outputs.append(sequence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, type=str, default='', help='Path to a checkpoint file (.pth)')
    opt = parser.parse_args()

    eval(opt.checkpoint)