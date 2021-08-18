1. Pre-training Cond-BART (ours: varying mask ratio)
    ./run_data_generation_gmodel.sh
    cd revised_libs/fairseq
    ./data_processing.sh
    
    Example 1) Pre-train on SNIPS-step
    fairseq-train SNIPS-step-GEN-bin --checkpoint-suffix _SNIPS_step_our --dataset SNIPS --data_setting step --restore-file /workspace/Imbalanced/nlp/data/model/bart.large/model.pt --max-tokens 512 --task denoising --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_large --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr 5e-05 --total-num-update 20000 --warmup-updates 500 --update-freq 4 --skip-invalid-size-inputs-valid-test --replace-length 1 --find-unused-parameters --rotate 0.0 --sample-break-mode 'eos' --min_mask 0.2 --max_mask 1.0 --mask-random 0.0 --mask-length 'word' --poisson-lambda 0.0 --valid-subset 'valid' --memory-efficient-fp16 --save-interval 20 --max-epoch 100;
    
    
2. Data generation
    Example 1) Augment SNIPS-longtail with CGS_d:
        python translation.py --gpu --device 0 --dataset SNIPS --data_setting step --cmodel our --gmodel our --imbalanced_ratio 100 --source_selection cluster --use_token_importance --random_seed 7777
    
    
3. Training Text Classification
    ./train_text_classification.sh [dataset] [data_setting] [cmodel] [gmodel]
    or
    run train.py with custom arguments
    
    Example 1) Train on TREC-longtail augmented by CGS_d:
        ./train_text_classification.sh TREC longtail our our

    Example 2) Train on TREC-step augmented by CSS_f:
        python train.py --num_of_epoch 50 --gpu --device 0 --TMix True --dataset TREC --data_setting step --train_bert --imbalanced_ratio 100 --random_seed 7777

    Example 3) Train on ATIS augmented by Cond-BART:
        python train.py --num_of_epoch 100 --gpu --device 0 --dataset $dataset --data_setting ATIS --data_augment --train_bert  --gmodel bart --imbalanced_ratio 100 --random_seed 7777

    Example 4) Train on SNIPS-step augmented by LAMBADA:
        ./train_text_classification.sh SNIPS step standard lambada

    Example 5) Train on TREC-step without augmentation:
        python train.py --num_of_epoch 100 --gpu --device 0 --dataset TREC --data_setting step --train_bert --imbalanced_ratio 100 --random_seed 7777


