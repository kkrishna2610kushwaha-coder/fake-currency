
import os
import gdown
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "currency_model.keras"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        file_id = "1faVpQ_RelHTYYjP8Ahqm4afEbYb0IEAQ"

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(
            url,
            str(MODEL_PATH),
            quiet=False,
            fuzzy=True
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model download failed!")

    return tf.keras.models.load_model(MODEL_PATH)
