import lightning as pl
import torch
import torch.utils
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem

# from main5 import prediction_results_app  # Import your prediction results Streamlit app
from model import GRUModel, TransformerModel, DNNSimpleModel, SmilesTrainDataset

gru_model = GRUModel.load_from_checkpoint(
    is_pretrain=False,
    checkpoint_path="torch_lt/gru_ft/lightning_logs/version_1/checkpoints/epoch=9-step=440.ckpt",
)
transformer_model = TransformerModel.load_from_checkpoint(
    is_pretrain=False,
    checkpoint_path="torch_lt/transfomer_ft/lightning_logs/version_4/checkpoints/epoch=9-step=440.ckpt",
)
dnn_model = DNNSimpleModel.load_from_checkpoint(
    is_pretrain=False,
    checkpoint_path="torch_lt/dnn_ft/lightning_logs/version_1/checkpoints/epoch=9-step=440.ckpt",
)

text_dataset = SmilesTrainDataset(
    data_path="torch_lt/data_dups_removed.csv", is_train=False
)

# Trainer configuration
trainer_transformer = pl.Trainer(default_root_dir="torch_lt/transfomer_test/")
trainer_dnn = pl.Trainer(default_root_dir="torch_lt/dnn_test/")
trainer_gru = pl.Trainer(default_root_dir="torch_lt/gru_test/")

val_dataloader = torch.utils.data.DataLoader(text_dataset, batch_size=1, shuffle=False)


def display_metrics(model_name, accuracy, auc, recall, prc_auc):
    st.write(f"Model: {model_name}")
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"AUC Score: {auc:.4f}")  # Display AUC
    st.write(f"Recall: {np.mean(recall):.4f}")  # Display average recall
    st.write(f"PRC AUC: {prc_auc:.4f}")
    st.write("---")


# Streamlit UI
def main():
    st.title("Tox21 Drug Effect Prediction")

    # Add a selectbox to switch between pages
    # model_type = st.sidebar.selectbox("select model", ["tox_only", "sider_tox"])
    page = st.sidebar.selectbox(
        "Select a page", ["Prediction Results", "Evaluation Metrics"]
    )

    # Page 1: Prediction Results
    if page == "Prediction Results":
        st.header("Prediction Results")

        # Input SMILES string
        smiles_input = st.text_input("Enter SMILES string:")

        if st.button("Predict"):
            # pass
            # Preprocess the input SMILES string
            mol = Chem.MolFromSmiles(smiles_input)
            if mol is not None:
                fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                input_tensor = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0)

                transformer_model.eval()
                with torch.no_grad():
                    # print(smiles_input)
                    pred = transformer_model(input_tensor.to(transformer_model.device))

                pred = pred.softmax(dim=1).cpu().numpy()
                print(pred)
                
                #     # Threshold for classification

                threshold = 0.5

                # #     # Classify predictions based on the threshold
                # #     gru_classification = (gru_pred > threshold).astype(int)
                # #     dnn_classification = (dnn_pred > threshold).astype(int)
                transformer_classification = (pred > threshold).astype(int)

                # #     # Display classifications in a table
                st.subheader("Classifications")
                table_data = {
                        'Model': ['Transformer'],
                        'NR-AR': [transformer_classification[0][0]],
                        'NR-AR-LBD': [transformer_classification[0][1]],
                        'NR-AhR': [transformer_classification[0][2]],
                        'NR-Aromatase': [transformer_classification[0][3]],
                        'NR-ER': [transformer_classification[0][4]],
                        'NR-ER-LBD': [transformer_classification[0][5]],
                        'NR-PPAR-gamma': [transformer_classification[0][6]],
                        'SR-ARE': [ transformer_classification[0][7]],
                        'SR-ATAD5': [transformer_classification[0][8]],
                        'SR-HSE': [transformer_classification[0][9]],
                        'SR-MMP': [transformer_classification[0][10]],
                        'SR-p53': [transformer_classification[0][11]]
                    }

                st.table(pd.DataFrame(table_data))
            else:
                st.error("Invalid SMILES string. Please enter a valid SMILES.")

    # Page 2: Evaluation Metrics and Figures
    elif page == "Evaluation Metrics":
        st.header("Evaluation Metrics and Figures")

        # Evaluate models
        gru_res = trainer_gru.validate(gru_model, dataloaders=val_dataloader)[0]
        # print(gru_res)
        gru_accuracy = gru_res["val_accuracy"]
        gru_auc = gru_res["val_precision"]
        gru_recall = gru_res["val_recall"]
        gru_prc_auc = gru_res["val_f1"]

        dnn_res = trainer_dnn.validate(dnn_model, dataloaders=val_dataloader)[0]
        # print(gru_res)
        dnn_accuracy = dnn_res["val_accuracy"]
        dnn_auc = dnn_res["val_precision"]
        dnn_recall = dnn_res["val_recall"]
        dnn_prc_auc = dnn_res["val_f1"]

        transformer_res = trainer_transformer.validate(
            transformer_model, dataloaders=val_dataloader
        )[0]
        # print(gru_res)
        transformer_accuracy = transformer_res["val_accuracy"]
        transformer_auc = transformer_res["val_precision"]
        transformer_recall = transformer_res["val_recall"]
        transformer_prc_auc = transformer_res["val_f1"]

        # Display evaluation metrics
        display_metrics("GRU", gru_accuracy, gru_auc, gru_recall, gru_prc_auc)
        display_metrics("DNN", dnn_accuracy, dnn_auc, dnn_recall, dnn_prc_auc)
        display_metrics(
            "Transformer",
            transformer_accuracy,
            transformer_auc,
            transformer_recall,
            transformer_prc_auc,
        )

        # Plot F1 scores along epochs for each model
        st.set_option("deprecation.showPyplotGlobalUse", False)

        gru_metric = pd.read_csv("torch_lt/gru_ft/lightning_logs/version_1/metrics.csv")
        dnn_metric = pd.read_csv("torch_lt/dnn_ft/lightning_logs/version_1/metrics.csv")
        transformer_metric = pd.read_csv("torch_lt/transfomer_ft/lightning_logs/version_4/metrics.csv")

        def load_train_metrics(metric_type="train_loss"):
            gru_losses = gru_metric[metric_type].values
            gru_losses = [item for item in gru_losses if not np.isnan(item)]
            print(len(gru_losses))
            print(gru_losses)

            dnn_losses = dnn_metric[metric_type].values
            dnn_losses = [item for item in dnn_losses if not np.isnan(item)]

            transformer_losses = transformer_metric[metric_type].values
            transformer_losses = [item for item in transformer_losses if not np.isnan(item)]

            return gru_losses, dnn_losses, transformer_losses

        gru_losses, dnn_losses, transformer_losses = load_train_metrics("train_loss")
        gru_accuracys, dnn_accuracys, transformer_accuracys = load_train_metrics("val_accuracy")
        gru_recalls, dnn_recalls, transformer_recalls = load_train_metrics("val_recall")
        gru_prc_aucs, dnn_prc_aucs, transformer_prc_aucs = load_train_metrics("val_f1")
        gru_aucs, dnn_aucs, transformer_aucs = load_train_metrics("val_precision")
        # Set custom figure size
        fig_size = (15, 10)

        # Plot training loss
        plt.figure(figsize=fig_size)
        plt.subplot(3, 2, 1)
        plt.plot(range(1, len(gru_losses) + 1), gru_losses, label='GRU',color='blue')
        plt.plot(range(1, len(dnn_losses) + 1), dnn_losses, label='DNN',color='orange')
        plt.plot(range(1, len(transformer_losses) + 1), transformer_losses, label='Transformer',color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Losses of Models')
        plt.legend()

        # Plot accuracy
        # plt.figure(figsize=fig_size)
        # Plot accuracy
        plt.subplot(3, 2, 2)
        plt.plot(range(1, len(gru_accuracys) + 1), gru_accuracys, label='GRU', color='blue')
        plt.plot(range(1, len(gru_accuracys) + 1), dnn_accuracys, label='DNN', color='orange')
        plt.plot(range(1, len(transformer_accuracys) + 1), transformer_accuracys, label='Transformer', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracies of Models')
        plt.legend()

        # Plot recall
        # plt.figure(figsize=fig_size)
        plt.subplot(3, 2, 3)
        plt.plot(range(1, len(gru_recalls) + 1), gru_recalls, label='GRU',color='blue')
        plt.plot(range(1, len(dnn_recalls) + 1), dnn_recalls, label='DNN',color='orange')
        plt.plot(range(1, len(transformer_recalls) + 1), transformer_recalls, label='Transformer',color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recalls of Models')
        plt.legend()

        # Plot PRC AUC
        # plt.figure(figsize=fig_size)
        # Plot PRC AUC
        plt.subplot(3, 2, 4)
        plt.plot(range(1, len(gru_prc_aucs) + 1), gru_prc_aucs, label='GRU',color='blue')
        plt.plot(range(1, len(dnn_prc_aucs) + 1), dnn_prc_aucs, label='DNN',color='orange')
        plt.plot(range(1, len(transformer_prc_aucs) + 1), transformer_prc_aucs, label='Transformer',color='green')
        plt.xlabel('Epoch')
        plt.ylabel('PRC AUC')
        plt.title('PRC AUCs of Models')
        plt.legend()

        # Plot AUC
        # plt.figure(figsize=fig_size)
        # Plot AUC
        # Plot PRC AUC
        # Plot AUC
        plt.subplot(3, 2, 5)
        plt.plot(range(1, len(gru_aucs) + 1), gru_aucs, label='GRU', color='blue')
        plt.plot(range(1, len(dnn_aucs) + 1), dnn_aucs, label='DNN', color='orange')
        plt.plot(range(1, len(transformer_aucs) + 1), transformer_aucs, label='Transformer', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('AUC of Models')
        plt.legend()

        plt.tight_layout()
        st.pyplot()


# Run the Streamlit app
if __name__ == "__main__":
    main()
