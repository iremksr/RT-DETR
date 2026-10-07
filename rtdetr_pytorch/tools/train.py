"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) #
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed() # Birden fazla gpu da distirbutd çalıştırmak için.
    if args.seed is not None:  # Randomized işlerde reproducibility için seed belirlenmesi gerek.
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # FROM SCRATCH
    # Modeli başlangıçtan eğit
    #         ↓
    # yeni training


    # RESUME
    # Aynı training'in checkpoint'ini yükle
    #         ↓
    # kaldığın yerden devam


    # TUNING
    # Önceden eğitilmiş ağırlıkları yükle
    #         ↓
    # yeni göreve/dataset'e yönelik
    # yeni training

    cfg = YAMLConfig(
        args.config, # Direkt config dosyasını alır.
        resume=args.resume, 
        use_amp=args.amp, #AMP (Automtic Mixed Precision): Yarı hassasiyetli (16 bit) ve tam hassasiyetli (32 bit) hesaplamaları birleştirerek eğitim sürecini hızlandıran bir teknik.
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, help='seed',)
    args = parser.parse_args()

    main(args)
