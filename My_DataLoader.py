import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import re

class Images(Dataset):
    """
    Dataset personalizzato per caricare immagini da una struttura di cartelle annidate,
    estrarre il ground truth dal nome del file e applicare trasformazioni sia per
    la versione RGB che per quella in scala di grigi.
    """
    def __init__(self, root_dir, rgb_transform=None, grayscale_transform=None):
        """
        Inizializza il dataset con la root directory e le trasformazioni per RGB e Grayscale.
        """
        self.root_dir = root_dir
        self.rgb_transform = rgb_transform
        self.grayscale_transform = grayscale_transform
        self.image_paths = []
        self.ground_truths = []

        # Scansiona le directory per trovare tutti i percorsi delle immagini e le loro ground truth
        self._scan_directories()

        if not self.image_paths:
            raise RuntimeError(f"Nessuna immagine trovata nella directory: {root_dir}")

    def _scan_directories(self):
        """
        Scansiona ricorsivamente la root_dir per trovare tutti i percorsi delle immagini
        e popolare le liste self.image_paths e self.ground_truths.
        """
        print(f"Scansione della directory: {self.root_dir}")
        # os.walk attraversa la directory ad albero (root, dirs, files)
        for dirpath, _, filenames in os.walk(self.root_dir):
            for filename in filenames:
                # Controlla se il file è un'immagine (puoi estendere questa lista)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    img_path = os.path.join(dirpath, filename)
                    try:
                        # Estrai la ground truth dal nome del file
                        ground_truth = self.ground_truth_extraction(filename)
                        self.image_paths.append(img_path)
                        self.ground_truths.append(ground_truth)
                    except ValueError as e:
                        print(f"Skipping file '{filename}' due to parsing error: {e}")
                    except Exception as e:
                        print(f"An unexpected error occurred with file '{filename}': {e}")


    def ground_truth_extraction(self, filename):
        """
        Estrae latitudine, longitudine e bearing da un nome di file con il formato specificato.

        Args:
            filename (str): Il nome del file da analizzare.

        Returns:
            tuple: Una tupla contenente (latitudine, longitudine, bearing in gradi) come float.

        Raises:
            ValueError: Se non è possibile estrarre i valori o se il bearing non è convertibile.
        """
        match = re.search(r'@(?:[^@]+)@(?:[^@]+)@(?:[^@]+)@(?:[^@]+)@(-?\d+\.?\d*)@(-?\d+\.?\d*)@(?:[^@]+)@(?:[^@]+)@(?:[^@]+)@(?:[^@]+)@(\d+)@(?:[^@]+)\.jpg$', filename, re.IGNORECASE)
        if match:
            try:
                latitude = float(match.group(1))
                longitude = float(match.group(2))
                # Assuming the bearing is the last numeric part before the final '@@.jpg' and needs conversion
                bearing = float(match.group(3)) # Assuming it's already in a unit that can be directly converted to degrees

                return (latitude, longitude, bearing)
            except ValueError as e:
                raise ValueError(f"Impossibile convertire i valori estratti in float per '{filename}': {e}")
        else:
            raise ValueError(f"Impossibile estrarre latitudine, longitudine e bearing dal nome del file: '{filename}'")

    def __len__(self):
        """
        Restituisce il numero totale di immagini nel dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx, color=True):
        """
        Restituisce il campione (immagine RGB, immagine in scala di grigi e ground truth)
        per l'indice dato.

        Args:
            idx (int): Indice del campione.

        Returns:
            tuple: Un tuple contenente (immagine_rgb, immagine_grayscale, ground_truth).
                   Le immagini saranno tensor di PyTorch e ground_truth sarà un tensor float.
        """
        img_path = self.image_paths[idx]
        ground_truth = self.ground_truths[idx]

        # Carica l'immagine usando PIL e convertila sempre in RGB inizialmente
        # Questo garantisce che entrambe le trasformazioni (RGB e Grayscale) partano dallo stesso formato base
        original_image = Image.open(img_path).convert('RGB')

        # Applica le trasformazioni
        if color:
            image = self.rgb_transform(original_image)
        else:
            image = self.grayscale_transform(original_image)

        # Converte la ground truth in un tensor di PyTorch
        ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)

        # Restituisce entrambe le versioni dell'immagine e la ground truth
        return image, ground_truth_tensor

# --- Definizione delle trasformazioni per immagini RGB ---
# Queste trasformazioni sono tipiche per i modelli pre-addestrati su ImageNet (3 canali).
rgb_transforms = transforms.Compose([
    transforms.Resize((256, 256)), # Ridimensiona l'immagine a una dimensione uniforme
    transforms.ToTensor(),         # Converte l'immagine PIL in un tensor PyTorch [C, H, W] e scala a [0.0, 1.0]
    # Normalizzazione con media e deviazione standard di ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Definizione delle trasformazioni per immagini in scala di grigi ---
# Queste trasformazioni includono la conversione a 1 canale e la normalizzazione specifica.
grayscale_transforms = transforms.Compose([
    transforms.Resize((256, 256)),      # Ridimensiona l'immagine
    transforms.Grayscale(num_output_channels=1), # CONVERTI L'IMMAGINE IN SCALA DI GRIGI (1 CANALE)
    transforms.ToTensor(),              # Converte in tensor PyTorch [1, H, W] e scala a [0.0, 1.0]
    # Normalizzazione per dati in scala di grigi scalati tra 0 e 1, spostando i valori a [-1, 1]
    transforms.Normalize(mean=[0.5], std=[0.5])
])
