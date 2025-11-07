import torch, cv2, numpy as np

class BehaviorVerifier:

    def __init__(self, weights_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # adapt if your model is torchscript vs state_dict
        self.model = torch.jit.load(weights_path, map_location=self.device) \
            if weights_path.endswith(".pt") else torch.load(weights_path, map_location=self.device)
        self.model.eval()

        self.size = 224  # EDIT if needed
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)
        # IMPORTANT: set order to match your training head
        self.labels = ["smoking", "vaping"]

    def _prep(self, bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(bgr, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2,0,1))
        t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return t

    @torch.inference_mode()
    def predict(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "smoking", 0.0  # default safe; won't flip if below thresh
        x = self._prep(crop_bgr)
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        probs = torch.softmax(out, dim=1).squeeze(0).detach().cpu().numpy()
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])
