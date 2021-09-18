import torch
import pytorch_lightning as pl

import nlp
import transformers


class IMDBSentiClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

        def _tokenize(x):
            # contains both text and encoded values
            x['input_ids'] = tokenizer.encode(
                x['text'],
                max_length=32,
                pad_to_max_length=True)

            return x

        def _prepare_ds(folder):
            ds = nlp.load_dataset('imdb', split=f'{folder}[:5%]')
            ds = ds.map(_tokenize)
            ds.set_format(type='torch', columns=['input_ids', 'label'])  # Maybe remove

            return ds

        self.train_ds, self.test_ds = map(_prepare_ds, ('train', 'test'))

    def forward(self, input_ids):
        mask = (input_ids != 0).float()
        logits, = self.model(input_ids, mask)

        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label']).mean()
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        acc = (logit.argmax(-1) == batch['label']).float()

        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([o['loss'] for o in outputs], 0).mean()
        acc = torch.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}

        return {**out, 'log': out}  # appending dic **

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=8,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=8,
            drop_last=False,
            shuffle=False
        )

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=1e-2,
            momentum=.9
        )


def main():
    model = IMDBSentiClassifier()

    trainer = pl.Trainer(
        default_root_dir='logs',
        gpus=(1 if torch.cuda.is_available() else 0),
        max_epochs=10,
        logger=pl.loggers.TensorBoardLogger('logs/', name='imdb', version=0)
    )

    trainer.fit(model)


main()