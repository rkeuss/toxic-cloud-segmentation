import torch
import warnings

class PseudoLabelGenerator:
    def __init__(
            self, teacher_model, threshold=0.5, device='cuda', dynamic_threshold=True,
            class_balanced=True, s=0.8, beta=0.9, class_threshold_ema=None
    ):
        self.teacher_model = teacher_model
        self.threshold = threshold
        self.device = device
        self.dynamic_threshold = dynamic_threshold
        self.class_balanced = class_balanced
        self.s = s
        self.beta = beta
        self.class_threshold_ema = class_threshold_ema

    def __call__(self, dataloader):
        """
        Generate pseudo-labels for unlabeled data using the teacher model's predictions.

        Args:
            teacher_model (torch.nn.Module): The trained teacher model.
            dataloader (torch.utils.data.DataLoader): DataLoader for unlabeled data.
            threshold (float): Confidence threshold for binary segmentation.
            device (str): Device to run the model on ('cuda' or 'cpu').
            dynamic_threshold (bool): Whether to use dynamic thresholding based on confidence distribution.
            class_balanced (bool): Whether to apply class-specific thresholds for balanced pseudo-labels.

        Yields:
            tuple: A batch of images and their corresponding pseudo-labels.
        """
        self.teacher_model.eval()
        for images, _ in dataloader:
            if images is None or images.size(0) == 0:
                warnings.warn("Empty or invalid batch, skipping.")
                continue
            images = images.to(self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    logits = self.teacher_model(images)
                probs = torch.softmax(logits, dim=1)
                confidence, pseudo_labels = torch.max(probs, dim=1)

                if self.dynamic_threshold and self.class_balanced:
                    class_thresholds = self.compute_class_thresholds_with_reservation(probs=probs, s=self.s)
                    self.class_threshold_ema = self.update_threshold_ema(class_thresholds, self.class_threshold_ema, beta=self.beta)

                    for c in range(probs.shape[1]):
                        mask = (pseudo_labels == c) & (confidence < self.class_threshold_ema[c])
                        pseudo_labels[mask] = 255
                else:
                    pseudo_labels[confidence < self.threshold] = 255

            yield images.cpu(), pseudo_labels.cpu()
            del logits, probs, confidence, pseudo_labels
            torch.cuda.empty_cache()
        self.teacher_model.train()

    @staticmethod
    def compute_class_thresholds_with_reservation(probs, s=0.8):
        thresholds = []
        for c in range(probs.shape[1]):
            class_conf = probs[:, c, :, :].flatten()
            if class_conf.numel() == 0:
                thresholds.append(0.0)
                continue
            k = int((1.0 - s) * class_conf.numel())
            threshold = torch.kthvalue(class_conf, k + 1).values.item() if k > 0 else class_conf.min().item()
            thresholds.append(threshold)
        return thresholds

    def update_threshold_ema(self, current_thresholds, ema_thresholds, beta=0.9):
        if ema_thresholds is None:
            return current_thresholds  # First iteration
        return [
            beta * old + (1 - beta) * new
            for old, new in zip(ema_thresholds, current_thresholds)
        ]

    def get_ema_thresholds(self):
        return self.class_threshold_ema
