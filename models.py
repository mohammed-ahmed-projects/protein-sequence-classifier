# models.py          
import os  
import torch  
import torch.nn as nn  
import matplotlib.pyplot as plt  
from io import BytesIO  
from Bio.SeqUtils.ProtParam import ProteinAnalysis  
  
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"  
aa_to_idx = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  # +1 for padding  
aa_to_idx["X"] = 0  # unknown / padding  
  
def preprocess(seq, max_len=1000):  
    if seq is None:  
        seq = ""  
    seq = "".join([aa if aa in aa_to_idx else "X" for aa in seq.upper()])  
    seq = seq[:max_len].ljust(max_len, "X")  
    return torch.tensor([aa_to_idx[aa] for aa in seq], dtype=torch.long)  
  
class ProteinCNN(nn.Module):  
    def __init__(self, vocab_size=21, embed_dim=50, num_classes=1):  
        super().__init__()  
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)  
        self.relu = nn.ReLU()  
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)  
        self.fc = nn.Linear(128, num_classes)  
  
    def forward(self, x):  
        x = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)  
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
  
class ProteinClassifier:  
    def __init__(self, cnn_path="protein_cnn_model.pth", lstm_path="protein_lstm_model.pth", device=None):  
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")  
  
        self.models = {  
            "CNN": ProteinCNN().to(self.device),  
            "LSTM": ProteinLSTM().to(self.device)  
        }  
  
        if os.path.exists(cnn_path):  
            try:  
                self.models["CNN"].load_state_dict(torch.load(cnn_path, map_location=self.device))  
                self.models["CNN"].eval()  
            except Exception as e:  
                print(f"[models.py] Warning: couldn't load CNN weights: {e}")  
  
        if os.path.exists(lstm_path):  
            try:  
                self.models["LSTM"].load_state_dict(torch.load(lstm_path, map_location=self.device))  
                self.models["LSTM"].eval()  
            except Exception as e:  
                print(f"[models.py] Warning: couldn't load LSTM weights: {e}")  
  
    def predict_sequence(self, seq, model_type="CNN"):  
        seq_tensor = preprocess(seq).unsqueeze(0).to(self.device)  
        model = self.models.get(model_type)  
        if model is None:  
            raise ValueError(f"Unknown model type: {model_type}")  
        model.eval()  
        with torch.no_grad():  
            out = model(seq_tensor)  
            prob = torch.sigmoid(out).item()  
        pred = "Enzyme" if prob > 0.5 else "Non-enzyme"  
        return pred, prob, self.analyze(seq), self.visualize(seq)  
  
    def analyze(self, seq):  
        if not seq:  
            return {"Error": "Empty sequence."}  
  
        seq = seq.upper()  
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
            "Hydrophobic ratio": round(hydrophobic/length, 3),  
            "Net charge": charge,  
            "Catalytic residues": "Yes" if catalytic else "No",  
            "Molecular weight (Da)": round(mw, 2) if mw else "N/A",  
            "Isoelectric point (pI)": round(pI, 2) if pI else "N/A",  
            "Polar ratio": round(polar, 3),  
            "Non-polar ratio": round(nonpolar, 3),  
            "Charged ratio": round(charged, 3),  
        }  
  
    def visualize(self, seq):  
        seq = seq or ""  
        fig, ax = plt.subplots(figsize=(8, 4))  
        counts = {aa: seq.count(aa) for aa in AMINO_ACIDS}  
        ax.bar(list(counts.keys()), list(counts.values()))  
        ax.set_title("Amino Acid Composition")  
        ax.set_ylabel("Count")  
        plt.tight_layout()  
        buf = BytesIO()  
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")  
        plt.close(fig)  
        buf.seek(0)  
        return buf.getvalue()