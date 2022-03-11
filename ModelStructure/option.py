class option():
    def __init__(self):
        self.model_path = "E:/ProjectData/NCLS/Beaver-AllData"
        self.vocab = ['D:/PythonProject/Study202203Restart/Pretreatment/SharedDictionary.vocab']
        self.layers = 6
        self.heads = 8
        self.hidden_size = 512
        self.ff_size = 2048
        self.batch_size = 1024
        self.warm_up = 16000
        self.report_every = 5000
        self.save_every = 5000
        self.max_to_keep = 50
        self.tf = True
        self.label_smoothing = 0.0
        self.dropout = 0.2
        self.lr = 1E-4
        self.grad_accum = 1
        self.beam_size = 1
        self.length_penalty = 0.5
        self.max_length = 64
        self.min_length = 16
