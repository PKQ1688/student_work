import lightning as pl
import torch
import torch.utils


from model import GRUModel, TransformerModel, DNNSimpleModel, SmilesTrainDataset

batch_size = 64
learning_rate = 1e-3
num_epochs = 10


# Model instantiation
gru_model = GRUModel.load_from_checkpoint(is_pretrain=False,checkpoint_path="torch_lt/gru/lightning_logs/version_0/checkpoints/epoch=9-step=3340.ckpt")
transformer_model = TransformerModel.load_from_checkpoint(is_pretrain=False,checkpoint_path="torch_lt/transfomer/lightning_logs/version_0/checkpoints/epoch=9-step=3340.ckpt")
dnn_model = DNNSimpleModel.load_from_checkpoint(is_pretrain=False,checkpoint_path="torch_lt/dnn/lightning_logs/version_0/checkpoints/epoch=9-step=3340.ckpt")

train_dataset = SmilesTrainDataset(data_path="torch_lt/data_dups_removed.csv",is_train=True)
text_dataset = SmilesTrainDataset(data_path="torch_lt/data_dups_removed.csv",is_train=False)

# Trainer configuration
trainer_transformer = pl.Trainer(
    max_epochs=num_epochs, devices="auto", default_root_dir="torch_lt/transfomer_ft/"
)
trainer_dnn = pl.Trainer(
    max_epochs=num_epochs, devices="auto", default_root_dir="torch_lt/dnn_ft/"
)
trainer_gru = pl.Trainer(
    max_epochs=num_epochs, devices="auto", default_root_dir="torch_lt/gru_ft/"
)

dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    text_dataset, batch_size=1, shuffle=False
)

# Training and evaluation
# trainer.fit(gru_model,data_loader)
trainer_transformer.fit(transformer_model,dataloader, val_dataloaders=val_dataloader)
# trainer_transformer.validate(transformer_model, val_dataloader)


trainer_dnn.fit(dnn_model, dataloader,val_dataloaders=val_dataloader)
# trainer_dnn.validate(dnn_model, val_dataloader)

trainer_gru.fit(gru_model, dataloader,val_dataloaders=val_dataloader)
# trainer_gru.validate(gru_model, val_dataloader)