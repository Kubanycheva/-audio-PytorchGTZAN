import torchaudio
from fastapi import FastAPI,HTTPException,UploadFile,File
import uvicorn
import torch
import torch.nn as nn
from pydantic.experimental.pipeline import transform
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf

import torch
import torch.nn as nn


class CheckMelodia(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8))
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.first(x)
    x = self.second(x)
    return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64,

)
max_len = 500

genres = torch.load('labels.pth')
index_to_label = {ind: lab for ind, lab in enumerate(genres)}


model = CheckMelodia()
model.load_state_dict(torch.load('model (1).pth',map_location=device))
model.to(device)
model.eval()


def change_audio(waveform,sr):
    if sr != 22050:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    if spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))

    return spec


audio_app = FastAPI()

@audio_app.post('/predict')
async def predict_audio(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Пустой файл')

        wf, sr = sf.read(io.BytesIO(data), dtype='float32')
        wf = torch.tensor(wf).T if wf.ndim > 1 else torch.tensor(wf)

        spec = change_audio(wf, sr).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(spec)
            pred_ind = torch.argmax(y_pred, dim=1).item()
            pred_class = index_to_label[pred_ind]

        return {"индекс": pred_ind, "жанр": pred_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(audio_app, host="127.0.0.1", port=8000)
