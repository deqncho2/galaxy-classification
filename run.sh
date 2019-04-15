# requires astropy, and the packages from the mlp environment from the last semester

# export LOTSS_DATA_DIR=/path/to/dataset # unconmment this
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --num_epochs 50 \
                                                                                     --num_layers 3 \
                                                                                     --dropout_rate 0.5 \
                                                                                     --learning_rate 0.0001 \
                                                                                     --batch_size 128 \
                                                                                     --dim_reduction_type max_pooling \
                                                                                     --image_height 128 \
                                                                                     --image_width 128 \
                                                                                     --dataset A
