import os
import glob
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from src.Audio_Model.TorchBiLSTM import BiLSTMClassifier
from src.Text_Model.BERT_CNN import BERTCNN
from src.Visual_Model.DDA3D import DDAMNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    mfccs = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    mfccs_padded = pad_sequence(mfccs, batch_first=True)
    labels = torch.stack(labels)
    return mfccs_padded, labels

visual_model = DDAMNet().to(device)
audio_model = BiLSTMClassifier(48, 64, 8, num_layers=2, dropout=0.0).to(device)
text_model = BERTCNN(64, [3, 4, 5], 0.0).to(device)

visual_model.load_state_dict(torch.load(r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Visual_Model\saved_models\saved_ddamfn_model.pth", map_location=device)["model_state_dict"])
audio_model.load_state_dict(torch.load(r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Audio_Model\saved_bilstm_model.pt", map_location=device))
text_model.load_state_dict(torch.load(r"C:\Users\singh\PycharmProjects\CSC413_AI_Project\src\Text_Model\saved_models\saved_bertcnn_model.pt", map_location=device))

visual_model.eval()
audio_model.eval()
text_model.eval()

root_data_dir = r"E:\CSC413_Data\FUSION_DATA"
actor_dirs = sorted(glob.glob(os.path.join(root_data_dir, "Actor_*")))

pt_files = []
for actor_dir in actor_dirs:
    label_dirs = sorted(glob.glob(os.path.join(actor_dir, "*")))
    for label_dir in label_dirs:
        label_str = os.path.basename(label_dir)
        # Convert label_str "01" -> 0, "02"->1, ...
        label = int(label_str) - 1
        files = glob.glob(os.path.join(label_dir, "*.pt"))
        for f in files:
            pt_files.append((f, label))

correct = 0
total = 0

with torch.no_grad():
    for f, true_label in pt_files:
        data = torch.load(f)  
        visual_tensor, audio_tensor, text_tensor = data[0], data[1], data[2]

        visual_tensor = visual_tensor.unsqueeze(0).to(device)  

        audio_tensor = torch.tensor(audio_tensor)
        audio_tensor = audio_tensor.transpose(0, 1)
        audio_input, audio_label = collate_fn([(audio_tensor, torch.tensor(true_label))])
        audio_tensor = audio_tensor.transpose(1, 0)
        audio_input = audio_input.to(device)  
        audio_label = audio_label.to(device)

        if text_tensor[0].dim() == 2:
            text_tensor[0] = text_tensor[0].unsqueeze(0)
        text_tensor_input_id = text_tensor[0].to(device)
        text_tensor_att_mask = text_tensor[1].to(device)
        text_tensor_input_id = text_tensor_input_id.unsqueeze(0)
        text_tensor_att_mask = text_tensor_att_mask.unsqueeze(0)

        v_logits = visual_model(visual_tensor)  
        v_probs = F.softmax(v_logits[0], dim=1)

        a_logits = audio_model(audio_input)  
        a_probs = F.softmax(a_logits, dim=1)

        t_logits = text_model(text_tensor_input_id, text_tensor_att_mask)  
        t_probs = F.softmax(t_logits, dim=1)

        a_probs = a_probs.squeeze(0)  
        v_probs = v_probs.squeeze(0)  
        t_probs = t_probs.squeeze(0) 

        numerator = a_probs * v_probs * t_probs  
        denominator = numerator.sum()
        fused_probs = numerator / denominator  

        pred_label = torch.argmax(fused_probs).item()
        if pred_label == true_label:
            correct += 1
        total += 1

accuracy = correct / total if total > 0 else 0.0
print(f"Accuracy: {accuracy * 100:.2f}%")
