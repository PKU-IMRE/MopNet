#!/usr/bin/env bash
python test.py --dataroot /media/he/FE2CA0442C9FF5BD/test  --netG 'mopnet/netG_epoch_150.pth' --netE "mopnet/netEdge_epoch_150.pth" --batchSize 2  --originalSize  256 --imageSize 256 --image_path "results" --write 1
