# models.py
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from io import BytesIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Amino acid vocabulary
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # +1 for padding
aa_to_idx["X"] = 0  # unknown / padding

def preprocess(seq, max_len=1000):
    if seq is None:
        seq = ""
    seq = "".join([aa if aa in aa_to_idx else "X" for aa in seq.upper()])
    seq = seq[:max_len].ljust(max_len, "X")
    return torch.tensor([aa_to_idx[aa] for aa in seq], dtype=torch.long)


# ------------------------
# Model definitions
# ------------------------
class ProteinCNN(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=50, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ProteinLSTM(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=50, hidden_dim=64, num_classes=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        return self.fc(hn)


# ------------------------
# Classifier wrapper
# ------------------------
class ProteinClassifier:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model file paths (same folder as this script)
        self.cnn_path = "protein_cnn_model.pth"
        self.lstm_path = "protein_lstm_model.pth"

        # Models
        self.models = {
            "CNN": ProteinCNN().to(self.device),
            "LSTM": ProteinLSTM().to(self.device),
        }
        self.models_loaded = {"CNN": False, "LSTM": False}

        # Load CNN
        self._load_model("CNN", self.cnn_path)

        # Load LSTM
        self._load_model("LSTM", self.lstm_path)

        # Warning if no models loaded
        if not any(self.models_loaded.values()):
            print("[models.py] WARNING: No models loaded. Using random weights!")

    def _load_model(self, model_type, path):
        if not os.path.exists(path):
            print(f"[models.py] {model_type} file not found: {path}")
            return

        print(f"[models.py] Trying to load {model_type} from {path} ...")
        try:
            # Try as state_dict
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self.models[model_type].load_state_dict(state_dict)
            self.models[model_type].eval()
            self.models_loaded[model_type] = True
            print(f"[models.py] Loaded {model_type} (state_dict)")
        except Exception as e1:
            print(f"[models.py] Failed to load {model_type} as state_dict: {e1}")
            try:
                # Try as full model
                self.models[model_type] = torch.load(path, map_location=self.device)
                self.models[model_type].eval()
                self.models_loaded[model_type] = True
                print(f"[models.py] Loaded {model_type} (full model)")
            except Exception as e2:
                print(f"[models.py] Failed to load {model_type}: {e2}")

    def predict_sequence(self, seq, model_type="CNN"):
        if not self.models_loaded.get(model_type, False):
            return (
                "Model Not Available",
                0.5,
                {"Error": f"{model_type} model not loaded."},
                None
            )

        try:
            seq_tensor = preprocess(seq).unsqueeze(0).to(self.device)
            model = self.models.get(model_type)
            if model is None:
                raise ValueError(f"Unknown model type: {model_type}")

            model.eval()
            with torch.no_grad():
                out = model(seq_tensor)
                prob = torch.sigmoid(out).item()

            pred = "Enzyme" if prob > 0.5 else "Non-enzyme"
            analysis = self.analyze(seq)

            return pred, prob, analysis, self.visualize(seq)

        except Exception as e:
            return (
                "Error",
                0.0,
                {"Error": f"Prediction failed: {str(e)}"},
                None
            )

    def analyze(self, seq):
        if not seq or seq.strip() == "":
            return {"Error": "Empty sequence provided."}

        seq = seq.upper().strip()
        seq = "".join([aa for aa in seq if aa in AMINO_ACIDS])
        if not seq:
            return {"Error": "No valid amino acids found."}

        length = len(seq)
        hydrophobic = sum(1 for aa in seq if aa in "AILMFWYV")
        charge = sum(1 for aa in seq if aa in "KR") - sum(1 for aa in seq if aa in "DE")
        catalytic = any(aa in seq for aa in "DEHKSTY")

        try:
            analysed = ProteinAnalysis(seq)
            mw = analysed.molecular_weight()
            pI = analysed.isoelectric_point()
        except Exception:
            mw, pI = None, None

        polar = sum(1 for aa in seq if aa in "STNQY") / length
        nonpolar = sum(1 for aa in seq if aa in "AVLIMFWPG") / length
        charged = sum(1 for aa in seq if aa in "KRHDE") / length

        return {
            "Length": length,
            "Hydrophobic ratio": round(hydrophobic / length, 3),
            "Net charge": charge,
            "Catalytic residues": "Yes" if catalytic else "No",
            "Molecular weight (Da)": round(mw, 2) if mw else "N/A",
            "Isoelectric point (pI)": round(pI, 2) if pI else "N/A",
            "Polar ratio": round(polar, 3),
            "Non-polar ratio": round(nonpolar, 3),
            "Charged ratio": round(charged, 3),
        }

    def visualize(self, seq):
        seq = seq.upper().strip() if seq else ""
        seq = "".join([aa for aa in seq if aa in AMINO_ACIDS])
        counts = {aa: seq.count(aa) for aa in AMINO_ACIDS}

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(counts.keys(), counts.values(), color="steelblue", alpha=0.7,
                      edgecolor="navy", linewidth=0.5)
        ax.set_title("Amino Acid Composition", fontsize=14, fontweight="bold")
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel("Amino Acid", fontsize=12)
        plt.xticks(rotation=45, ha="right")

        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f"{int(height)}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
