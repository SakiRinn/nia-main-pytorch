import utils.getter as getter
import utils.fileIO as fileIO
import utils.mask as mask
from modules.optimizer import TransformerLR
from datasets import ResDataset

import logging
from tqdm import trange

from torch.utils.data import DataLoader


logging.basicConfig(filename='train.log',
                    level=logging.DEBUG,
                    format='[%(levelname)s] %(asctime)s: %(message)s',
                    datefmt='%y-%b-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train(resume=''):
    # Config
    model_cfg = fileIO.load_config('./configs/model.yaml')
    run_cfg = fileIO.load_config('./configs/run.yaml')
    device = run_cfg['device']
    end_epoch = run_cfg['train']['epochs']

    # Dataset
    dataset = ResDataset()
    data_loader = DataLoader(dataset,
                             batch_size=run_cfg['train']['batch_size'],
                             num_workers=4,
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

    # Resume
    start_epoch = 0
    if resume != '':
        checkpoint, start_epoch = fileIO.find_checkpoint(resume)
        model.load_state_dict(checkpoint)

    # Module
    loss_fn = getter.get_loss(run_cfg['train']['loss'])
    optimizer = getter.get_optimizer(run_cfg['train']['optimizer'], model.parameters(), run_cfg['train']['lr'])
    if model_name == 'Transformer':
        lr_scheduler = TransformerLR(optimizer, model_cfg['Transformer']['embedding_dim'])

    # Main
    for epoch in trange(start_epoch + 1, end_epoch + 1):
        # Step
        for step, data in enumerate(data_loader, start=1):
            src, trg = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            if model_name == 'Seq2Seq':
                loss = loss_fn(model(src, trg, teacher_forcing_ratio), trg)
            elif model_name == 'Transformer':
                src_key_padding_mask = mask.key_padding_mask(src).to(device)
                trg_key_padding_mask = mask.key_padding_mask(trg).to(device)
                attn_mask = mask.square_subsequent_mask(trg.size(1)).to(device)
                loss = loss_fn(model(src, trg, src_key_padding_mask, trg_key_padding_mask, attn_mask),
                               trg, trg_key_padding_mask)

            loss.backward()
            if step % run_cfg['train']['log_interval_steps'] == 0 or step == len(data_loader):
                logger.info(f'[Epoch {epoch}/Step {step}] loss: {loss:.6f}, lr: {lr_scheduler.get_last_lr()[0]:.6f}')
            optimizer.step()
            lr_scheduler.step()

        # Save
        if epoch % run_cfg['train']['saving_interval_epochs'] == 0:
            fileIO.save_checkpoint(run_cfg['checkpoint_dir'], model, epoch)


if __name__ == "__main__":
    train()
