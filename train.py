import utils.getter as getter
import utils.fileIO as fileIO
import utils.mask as mask
from modules.optimizer import TransformerLR
from modules.loss import MaskedLoss
from datasets import ResDataset

from tqdm import trange

from torch.utils.data import DataLoader


def train(resume=''):

    # Config
    model_cfg = fileIO.load_config('./configs/model.yaml')
    run_cfg = fileIO.load_config('./configs/run.yaml')

    logger, exp_id = fileIO.init_logger(run_cfg['checkpoint_dir'])
    device = run_cfg['device']
    end_epoch = run_cfg['train']['epochs']
    lr = run_cfg['train']['lr']['initial']
    lr_scheduler = run_cfg["train"]['lr']['scheduler']

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
    if model_name == 'Transformer':
        loss_fn = MaskedLoss(loss_fn)
    optimizer = getter.get_optimizer(run_cfg['train']['optimizer'], model.parameters(), lr)
    lr_scheduler = getter.get_lr_scheduler(lr_scheduler, optimizer, model_cfg[model_name]['embedding_dim'])

    # Main
    for epoch in trange(start_epoch + 1, end_epoch + 1):
        # Step
        for step, data in enumerate(data_loader, start=1):
            src, trg = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            if model_name == 'Seq2Seq':
                out = model(src, trg, teacher_forcing_ratio)
                loss = loss_fn(out, trg)
            elif model_name == 'Transformer':
                src_key_padding_mask = mask.key_padding_mask(src).to(device)
                trg_key_padding_mask = mask.key_padding_mask(trg).to(device)
                attn_mask = mask.square_subsequent_mask(trg.size(1)).to(device)
                out = model(src, trg, src_key_padding_mask, trg_key_padding_mask, attn_mask)
                loss = loss_fn(out, trg, trg_key_padding_mask)
            else:
                out = model(src, trg)
                loss = loss_fn(out, trg)

            loss.backward()
            if step % run_cfg['train']['log_interval_steps'] == 0 or step == len(data_loader):
                logger.info(f'[Epoch {epoch}/Step {step}] loss: {loss:.6f}, lr: {lr_scheduler.get_last_lr()[0]:.6f}')
            optimizer.step()
            lr_scheduler.step()

        # Save
        if epoch % run_cfg['train']['saving_interval_epochs'] == 0:
            fileIO.save_checkpoint(run_cfg['checkpoint_dir'], model, epoch, exp_id=exp_id)


if __name__ == "__main__":
    train()
