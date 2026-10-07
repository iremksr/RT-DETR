"""by lyuwenyu
"""

from pprint import pprint
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler

from typing import Callable, List, Dict


__all__ = ['BaseConfig', ]



class BaseConfig(object):
    # TODO property


    def __init__(self) -> None:
        super().__init__()

        self.task :str = None 
        
        self._model :nn.Module = None # _model değişkenini ileride bir nn.Module yani  PyTorch sinir ağı modülü olması bekleniyor.
        self._postprocessor :nn.Module = None # Post-processing için kullanılan modül.
        self._criterion :nn.Module = None # Hatayı hesaplamak için kullanılan loss fonksiyonu.
        self._optimizer :Optimizer = None  # Optimizer: Modelin parametrelerini güncellemek için kullanılan algoritma.
        self._lr_scheduler :LRScheduler = None # LR (Learning Rate) Scheduler: Öğrenme oranını yönetmek için kullanılan modül.
        self._train_dataloader :DataLoader = None # Eğitim veri yükleyici.
        self._val_dataloader :DataLoader = None # Validation veri yükleyici.
        self._ema :nn.Module = None # Exponential Moving Average (EMA): Modelin ağırlıklarının zamanla ortalamasını almak için kullanılan bir teknik.
        self._scaler :GradScaler = None # GradScaler: AMP (Automatic Mixed Precision) kullanırken, gradyanları ölçeklendirmek için kullanılan bir modül.

        self.train_dataset :Dataset = None # Eğitim veri kümesi.
        self.val_dataset :Dataset = None # Validation veri kümesi.
        self.num_workers :int = 0 # Paralel işleme için kullanılacak iş parçacığı sayısı.
        self.collate_fn :Callable = None # Veri yükleyici için özel bir veri toplama (collate) fonksiyonu.

        self.batch_size :int = None # Genel batch boyutu.
        self._train_batch_size :int = None # Eğitim için özel batch boyutu.
        self._val_batch_size :int = None # Validation için özel batch boyutu.
        self._train_shuffle: bool = None # Eğitim veri kümesi için karıştırma (shuffle) işlemi.
        self._val_shuffle: bool = None # Validation veri kümesi için karıştırma (shuffle) işlemi.

        self.evaluator :Callable[[nn.Module, DataLoader, str], ] = None # Değerlendirme fonksiyonu. 

        # runtime
        self.resume :str = None # Devam edilen eğitimi yüklemek için kullanılan dosya yolu.
        self.tuning :str = None # Düzenleme için kullanılan dosya yolu.

        self.epoches :int = None # Toplam eğitim epoch sayısı.
        self.last_epoch :int = -1 # Son tamamlanan epoch sayısı. 
        self.end_epoch :int = None # Son epoch sayısı.

        self.use_amp :bool = False # Automatic Mixed Precision (AMP) kullanılıp kullanılmayacağını belirten bayrak.
        self.use_ema :bool = False # Exponential Moving Average (EMA) kullanılıp kullanılmayacağını belirten bayrak.
        self.sync_bn :bool = False # Synchronized Batch Normalization (SyncBN) kullanılıp kullanılmayacağını belirten bayrak.
        self.clip_max_norm : float = None # Gradyanların maksimum normunu sınırlamak için kullanılan değer. Bu, gradyan patlamasını önlemek için kullanılır.
        self.find_unused_parameters :bool = None # Unused parameters (kullanılmayan parametreler) olup olmadığını kontrol etmek için kullanılan bayrak. Bu, modelin bazı parametrelerinin kullanılmadığı durumlarda hata ayıklamaya yardımcı olabilir.
        # self.ema_decay: float = 0.9999
        # self.grad_clip_: Callable = None

        self.log_dir :str = './logs/' # Log dosyalarının kaydedileceği dizin.
        self.log_step :int = 10 # Loglama adımlarının sıklığı. Her log_step adımında bir log kaydı yapılır.
        self._output_dir :str = None # Çıktı dosyalarının kaydedileceği dizin. Eğer belirtilmezse log_dir kullanılır.
        self._print_freq :int = None # Yazdırma sıklığı. Her _print_freq adımında bir yazdırma yapılır.
        self.checkpoint_step :int = 1 # Checkpoint kaydetme adımlarının sıklığı. Her checkpoint_step adımında bir checkpoint kaydedilir.

        # self.device :str = torch.device('cpu')
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

        self.device = torch.device(device)

        print(f'Using device: {self.device}')


    @property
    def model(self, ) -> nn.Module:
        return self._model 
    
    @model.setter
    def model(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your model class'
        self._model = m 

    @property
    def postprocessor(self, ) -> nn.Module:
        return self._postprocessor
    
    @postprocessor.setter
    def postprocessor(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your model class'
        self._postprocessor = m 

    @property
    def criterion(self, ) -> nn.Module:
        return self._criterion
    
    @criterion.setter
    def criterion(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your model class'
        self._criterion = m 

    @property
    def optimizer(self, ) -> Optimizer:
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, m):
        assert isinstance(m, Optimizer), f'{type(m)} != optim.Optimizer, please check your model class'
        self._optimizer = m 

    @property
    def lr_scheduler(self, ) -> LRScheduler:
        return self._lr_scheduler
    
    @lr_scheduler.setter
    def lr_scheduler(self, m):
        assert isinstance(m, LRScheduler), f'{type(m)} != LRScheduler, please check your model class'
        self._lr_scheduler = m 


    @property
    def train_dataloader(self):
        if self._train_dataloader is None and self.train_dataset is not None:
            loader = DataLoader(self.train_dataset, 
                                batch_size=self.train_batch_size, 
                                num_workers=self.num_workers, 
                                collate_fn=self.collate_fn,
                                shuffle=self.train_shuffle, )
            loader.shuffle = self.train_shuffle
            self._train_dataloader = loader

        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        self._train_dataloader = loader 

    @property
    def val_dataloader(self):
        if self._val_dataloader is None and self.val_dataset is not None:
            loader = DataLoader(self.val_dataset, 
                                batch_size=self.val_batch_size, 
                                num_workers=self.num_workers, 
                                drop_last=False,
                                collate_fn=self.collate_fn, 
                                shuffle=self.val_shuffle)
            loader.shuffle = self.val_shuffle
            self._val_dataloader = loader

        return self._val_dataloader
    
    @val_dataloader.setter
    def val_dataloader(self, loader):
        self._val_dataloader = loader 


    # TODO method
    # @property
    # def ema(self, ) -> nn.Module:
    #     if self._ema is None and self.use_ema and self.model is not None:
    #         self._ema = ModelEMA(self.model, self.ema_decay)
    #     return self._ema

    @property
    def ema(self, ) -> nn.Module:
        return self._ema 

    @ema.setter
    def ema(self, obj):
        self._ema = obj
    

    @property
    def scaler(self) -> GradScaler: 
        if self._scaler is None and self.use_amp and torch.cuda.is_available():
            self._scaler = GradScaler()
        return self._scaler
    
    @scaler.setter
    def scaler(self, obj: GradScaler):
        self._scaler = obj


    @property
    def val_shuffle(self):
        if self._val_shuffle is None:
            print('warning: set default val_shuffle=False')
            return False
        return self._val_shuffle

    @val_shuffle.setter
    def val_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._val_shuffle = shuffle

    @property
    def train_shuffle(self):
        if self._train_shuffle is None:
            print('warning: set default train_shuffle=True')
            return True
        return self._train_shuffle

    @train_shuffle.setter
    def train_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._train_shuffle = shuffle


    @property
    def train_batch_size(self):
        if self._train_batch_size is None and isinstance(self.batch_size, int):
            print(f'warning: set train_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._train_batch_size

    @train_batch_size.setter
    def train_batch_size(self, batch_size):
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._train_batch_size = batch_size

    @property
    def val_batch_size(self):
        if self._val_batch_size is None:
            print(f'warning: set val_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._val_batch_size

    @val_batch_size.setter
    def val_batch_size(self, batch_size):
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._val_batch_size = batch_size


    @property
    def output_dir(self):
        if self._output_dir is None:
            return self.log_dir
        return self._output_dir

    @output_dir.setter
    def output_dir(self, root):
        self._output_dir = root

    @property
    def print_freq(self):
        if self._print_freq is None:
            # self._print_freq = self.log_step
            return self.log_step
        return self._print_freq

    @print_freq.setter
    def print_freq(self, n):
        assert isinstance(n, int), 'print_freq must be int'
        self._print_freq = n


    # def __repr__(self) -> str:
    #     pass 



