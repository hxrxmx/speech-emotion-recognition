import lightning

class SpeechEmotionsClassifier(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = None
        