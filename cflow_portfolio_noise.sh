# python3 end2end_cflowdfl_portfolio_alpha.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 40 --deg 4 --num_experiments 5 --alpha 1.0
# python3 end2end_cflowdfl_portfolio_alpha.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 60 --deg 4 --num_experiments 5 --alpha 1.0
# python3 end2end_cflowdfl_portfolio_alpha.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 80 --deg 4 --num_experiments 5 --alpha 1.0
# python3 end2end_cflowdfl_portfolio_alpha.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 100 --deg 4 --num_experiments 5 --alpha 1.0

python3 end2end_cflowdfl_portfolio_cvar.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 40 --deg 4 --num_experiments 5 --alpha 1.0 --risk True
python3 end2end_cflowdfl_portfolio_cvar.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 60 --deg 4 --num_experiments 5 --alpha 1.0 --risk True
python3 end2end_cflowdfl_portfolio_cvar.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 80 --deg 4 --num_experiments 5 --alpha 1.0 --risk True
python3 end2end_cflowdfl_portfolio_cvar.py --betas 100 --n 400 --m 50 --num_epochs 10 --noise_width 100 --deg 4 --num_experiments 5 --alpha 1.0 --risk True
