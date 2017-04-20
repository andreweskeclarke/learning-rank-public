python ranknet.py --lr 0.001 --batch=64 --n_hidden=20 --n_layers 1 --unfactorized # tony
python ranknet.py --lr 0.001 --batch=64 --n_hidden=20 --n_layers 2 --unfactorized # andrew
python ranknet.py --lr 0.001 --batch=64 --n_hidden=20 --n_layers 3 --unfactorized # joe
python ranknet.py --lr 0.001 --batch=64 --n_hidden=100 --n_layers 1 --unfactorized # tony
python ranknet.py --lr 0.001 --batch=64 --n_hidden=100 --n_layers 2 --unfactorized # andrew

python ranknet.py --lr 0.001 --batch=64 --n_hidden=20 --n_layers 1 --factorized # joe
python ranknet.py --lr 0.001 --batch=64 --n_hidden=20 --n_layers 2 --factorized # tony
python ranknet.py --lr 0.001 --batch=64 --n_hidden=20 --n_layers 3 --factorized # andrew
python ranknet.py --lr 0.001 --batch=64 --n_hidden=100 --n_layers 1 --factorized # joe
python ranknet.py --lr 0.001 --batch=64 --n_hidden=100 --n_layers 2 --factorized # tony

python ranknet.py --lr 0.00001 --batch=64 --n_hidden=20 --n_layers 1 --lambdarank # andrew
python ranknet.py --lr 0.00001 --batch=64 --n_hidden=20 --n_layers 2 --lambdarank # joe
python ranknet.py --lr 0.00001 --batch=64 --n_hidden=20 --n_layers 3 --lambdarank # tony
python ranknet.py --lr 0.00001 --batch=64 --n_hidden=100 --n_layers 1 --lambdarank # andrew
python ranknet.py --lr 0.00001 --batch=64 --n_hidden=100 --n_layers 2 --lambdarank # joe
