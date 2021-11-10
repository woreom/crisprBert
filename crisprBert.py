import torch
import argparse
from pathlib import Path
from transformers import RobertaConfig
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizerFast
from transformers import LineByLineTextDataset
from tokenizers.processors import BertProcessing
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from tokenizers.implementations import ByteLevelBPETokenizer

print("torch cuda is available: ", torch.cuda.is_available())


class Args():
    def __init__(self):
        self.data_path = "/mnt/d/M3/Projects/BCB/crisprBert/unlabeled_sgrna_fixed.txt"
        self.config_path = "/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/config/"
        self.output_path = "/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/"
        self.make_configs = True
        self.vocab_size = 52_000
        self.max_position_embeddings = 514 
        self.num_attention_heads = 12 
        self.num_hidden_layers = 6 
        self.type_vocab_size = 1 
        self.block_size = 128
        self.mlm =True
        self.mlm_probability = 0.025
        self.overwrite_output_dir = True
        self.num_train_epochs = 50 
        self.per_gpu_train_batch_size = 8
        self.per_gpu_eval_batch_size = 6
        self.learning_rate = 4e-4 
        self.weight_decay = 0.01 
        self.adam_beta1 = 0.9 
        self.adam_beta2 = 0.98 
        self.adam_epsilon = 1e-6 
        self.max_grad_norm = 1.0  
        self.logging_dir = "Models/crisprBert/log.txt" 
        self.logging_steps = 1000
        self.eval_steps = 10000
        self.save_steps = 10000 
        self.metric_for_best_model = "eval_loss" 
        self.save_total_limit = 10 
        self.prediction_loss_only =False
        self.do_train = True
        self.do_eval = True
        self.logging_first_step = True
        self.logging_nan_inf_filter = False
        self.greater_is_better = False
        self.load_best_model_at_end = True
        
    def init_parsearges(self):
        parser = argparse.ArgumentParser()
        # Required parameters
        parser.add_argument( "--data_path",
            default="/mnt/d/M3/Projects/BCB/crisprBert/unlabeled_sgrna_fixed.txt",
            type=str,
            required=True,
            help="The input data dir. Should contain the .txt file for the task.",
        )

        parser.add_argument( "--config_path",
            default="/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/config/",
            type=str,
            required=True,
            help="The config data dir. Should contain the vocab.json, merges.txt, config.json.",
        )

        parser.add_argument( "--output_path",
            default="/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/",
            type=str,
            required=True,
            help="The path to save the model.",
        )

        # Other parameters
        parser.add_argument( "--make_configs",
            action="store_true",
            help="Whether to make config and token file or read from dir",
        )
        parser.add_argument( "--vocab_size",
            default=52_000,
            type=int,
            help="size of the vocabulary",
        )
        parser.add_argument( "--max_position_embeddings",
            default=514,
            type=int,
            help="The maximum number of embeddings positions",
            )
        parser.add_argument( "--num_attention_heads",
            default=12,
            type=int,
            help="The number of attention head layers",
            )
        parser.add_argument( "--num_hidden_layers",
            default=6,
            type=int,
            help="The number of hidden layers",
            )
        parser.add_argument( "--type_vocab_size",
            default=1,
            type=int,
            help="ToDo",
            )

        parser.add_argument( "--block_size",
            default=128,
            type=int,
            help="ToDo",
            )
        parser.add_argument( "--mlm",
            action="store_true",
            help="ToDo",
            )
        parser.add_argument( "--mlm_probability",
            default=0.025,
            type=float,
            help="ToDo",
            )
        parser.add_argument( "--overwrite_output_dir",
            action="store_true",
            help="whether to overwrite existing models",
            )
        parser.add_argument( "--num_train_epochs",
            default=50,
            type=int,
            help="The number of training epochs",
            )
        parser.add_argument( "--per_gpu_train_batch_size",
            default=8,
            type=int,
            help="training batch size for gpu.",
            )

        parser.add_argument( "--per_gpu_eval_batch_size",
            default=6,
            type=int,
            help="evaluation batch size for gpu.",
            )

        parser.add_argument( "--learning_rate",
            default=0.01,
            type=float,
            help="learning rate.",
            )

        parser.add_argument( "--weight_decay",
            default=0.01,
            type=float,
            help="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer.",
            )

        parser.add_argument( "--adam_beta1",
            default=0.9,
            type=float,
            help="The beta1 hyperparameter for the AdamW optimizer.",
            )

        parser.add_argument(
            "--adam_beta2",
            default=0.98,
            type=float,
            help="The beta2 hyperparameter for the AdamW optimizer.",
            )

        parser.add_argument(
            "--adam_epsilon",
            default=1e-6,
            type=float,
            help="The epsilon hyperparameter for the AdamW optimizer.",
            )

        parser.add_argument(
            "--max_grad_norm",
            default=1.0,
            type=float,
            help="Maximum gradient norm (for gradient clipping).",
            )

        parser.add_argument(
            "--warmup_steps",
            default=100,
            type=int,
            help="Number of steps used for a linear warmup from 0 to learning_rate. Overrides any effect of warmup_ratio.",
            )

        parser.add_argument(
            "--logging_dir",
            default="Models/crisprBert/log.txt",
            type=str,
            help=" TensorBoard log directory. Will default to output_dir/runs/**CURRENT_DATETIME_HOSTNAME**.",
            )

        parser.add_argument(
            "--logging_steps",
            default=10_00,
            type=int,
            help=" Number of update steps between two logs if logging_strategy='steps'",
            )

        parser.add_argument(
            "--eval_steps",
            default=10_000,
            type=int,
            help="Evaluation is done (and logged) every eval_steps",
            )

        parser.add_argument(
            "--save_steps",
            default=10_000,
            type=int,
            help="The number of steps between checkpoints.",
            )

        parser.add_argument(
            "--metric_for_best_model",
            default='eval_loss',
            type=str,
            help="""Use in conjunction with load_best_model_at_end to specify the metric to use to compare two different models. Must be the name of a metric returned by the evaluation with or without the prefix "eval_". Will default to "loss" if unspecified and load_best_model_at_end=True (to use the evaluation loss).
        If you set this value, greater_is_better will default to True. Don’t forget to set it to False if your metric is better when lower.""",
            )

        parser.add_argument(
            "--save_total_limit",
            default=10,
            type=int,
            help="If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir",
            )
        parser.add_argument(
            "--prediction_loss_only",
            action="store_true",
            help="When performing evaluation and generating predictions, only returns the loss.",
            )
        parser.add_argument(
            "--do_train",
            action="store_true",
            help="Whether to run training or not. This argument is not directly used by Trainer, it’s intended to be used by your training/evaluation scripts instead. See the example scripts for more details.",
            )
        parser.add_argument(
            "--do_eval",
            action="store_true",
            help="Whether to run evaluation on the validation set or not. Will be set to True if evaluation_strategy is different from 'no'. This argument is not directly used by Trainer, it’s intended to be used by your training/evaluation scripts instead. See the example scripts for more details",
            )
        parser.add_argument(
            "--logging_first_step",
            action="store_true",
            help="Whether to log and evaluate the first global_step or not.",
            )
        parser.add_argument(
            "--logging_nan_inf_filter",
            action="store_true",
            help="Whether to filter nan and inf losses for logging. If set to obj:True the loss of every step that is nan or inf is filtered and the average loss of the current logging window is taken instead.",
            )

        parser.add_argument(
            "--greater_is_better",
            action="store_true",
            help="""Use in conjunction with load_best_model_at_end and metric_for_best_model to specify if better models should have a greater metric or not. Will default to:

        True if metric_for_best_model is set to a value that isn’t 'loss' or 'eval_loss'.

        False if metric_for_best_model is not set, or set to "='loss' or 'eval_loss'.""",
            )

        parser.add_argument(
            "--load_best_model_at_end",
            action="store_true",
            help="Whether or not to load the best model found during training at the end of training.",
            )

        args = parser.parse_args()
            
        return args


def make_token(paths:list=["/mnt/d/M3/Projects/BCB/crisprBert/unlabeled_sgrna_fixed.txt"], output_dir:str ="/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/configs/") -> None:
    print(paths)


    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])



    tokenizer.save_model(output_dir)

    

def make_config(vocab_size:int=52_000,max_position_embeddings:int=514,num_attention_heads:int=12,
 num_hidden_layers:int=6, type_vocab_size:int=1,output_dir:str="/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/configs/", save = True) -> RobertaConfig:

    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=type_vocab_size,
        )

    if save: config.to_json_file(output_dir+"config.json")
    return config



def check_configs(path:str="/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/configs/") -> None:



    tokenizer = ByteLevelBPETokenizer( path+"vocab.json", path+"merges.txt", )



    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)



    print(tokenizer.encode("CGCCGCCGCTTTCGGTGATGAGG"))


    print(tokenizer.encode("CGCCGCCGCTTTCGGTGATGAGG").tokens)


def load_model(config:RobertaConfig ,path:str="/mnt/d/M3/Projects/BCB/crisprBert/Models/crisprBert/configs/", ):

    tokenizer = RobertaTokenizerFast.from_pretrained(path, max_len=512)

    model = RobertaForMaskedLM(config=config)

    return model, tokenizer


def prepare_training(config:RobertaConfig, tokenizer:RobertaTokenizerFast, model:RobertaForMaskedLM, 
    file_path:str="./unlabeled_sgrna_fixed.txt", block_size:int=128, mlm:bool=True, mlm_probability:float=0.15, output_dir:str="Models/crisprBert/",
    overwrite_output_dir:bool=True, num_train_epochs:int=1, save_steps:int=10_000, save_total_limit:int=2,
    prediction_loss_only:bool=True, do_train:bool= False, do_eval:bool= False, do_predict:bool= False, per_gpu_train_batch_size:int = 8,
    per_gpu_eval_batch_size:int=6, learning_rate:float= 5e-05, weight_decay:float=0.0, adam_beta1:float=0.9, adam_beta2:float=0.999, adam_epsilon:float=1e-08,
    max_grad_norm:float=1.0, warmup_steps:int=0, logging_dir:str= "Models/crisprBert/log.txt", logging_first_step = True, logging_steps:int = 10_000, logging_nan_inf_filter:bool=True, fp16:bool=True,
    eval_steps:int=10_000,  disable_tqdm:bool= False, load_best_model_at_end:bool= True, metric_for_best_model:str="eval_loss", greater_is_better:bool = True,) -> Trainer:

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./unlabeled_sgrna_fixed.txt",
        block_size=128,
    )



    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        num_train_epochs=num_train_epochs,
        per_gpu_train_batch_size=per_gpu_train_batch_size,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        prediction_loss_only=prediction_loss_only, 
        do_train = do_train, 
        do_eval = do_eval,
        do_predict = do_predict,
        per_gpu_eval_batch_size = per_gpu_eval_batch_size,
        learning_rate = learning_rate,
        weight_decay = weight_decay, 
        adam_beta1 = adam_beta1,
        adam_beta2 = adam_beta2,
        adam_epsilon = adam_epsilon,
        max_grad_norm = max_grad_norm,  
        warmup_steps = warmup_steps,
        logging_dir = logging_dir,
        logging_first_step = logging_first_step,
        logging_steps = logging_steps,
        logging_nan_inf_filter = logging_nan_inf_filter, 
        fp16 = fp16, 
        eval_steps = eval_steps,  
        disable_tqdm = disable_tqdm, 
        load_best_model_at_end = load_best_model_at_end, 
        metric_for_best_model = metric_for_best_model,
        greater_is_better = greater_is_better, )


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )


    return trainer


def main(args):
    
    if args.make_configs:
        make_token(path=[args.data_path], output_dir=args.config_path)

    config = make_config(vocab_size=args.vocab_size,max_position_embeddings=args.max_position_embeddings, num_attention_heads=args.num_attention_heads,
            num_hidden_layers=args.num_hidden_layers, type_vocab_size=args.type_vocab_size,
            output_dir=args.config_path, save = args.make_configs)


    check_configs(path=args.config_path)

    model, tokenizer = load_model(config=config ,path=args.config_path, )

    trainer = prepare_training(config=config, tokenizer=tokenizer, model=model, 
    file_path=args.data_path, block_size=args.block_size, mlm=args.mlm, mlm_probability=args.mlm_probability, output_dir=args.output_path,
    overwrite_output_dir=args.overwrite_output_dir, num_train_epochs=args.num_train_epochs, per_gpu_train_batch_size=args.per_gpu_train_batch_size, save_steps=args.save_steps,
    save_total_limit=args.save_total_limit, prediction_loss_only=args.prediction_loss_only, do_train= args.do_train, do_eval=args.do_train,
    per_gpu_eval_batch_size=args.per_gpu_eval_batch_size, learning_rate = args.learning_rate, weight_decay=args.weight_decay, adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
    adam_epsilon=args.adam_epsilon, max_grad_norm=args.max_grad_norm, warmup_steps=args.warmup_steps, logging_dir=args.logging_dir, logging_first_step = args.logging_first_step,
    logging_steps = args.logging_steps, logging_nan_inf_filter=args.logging_nan_inf_filter, fp16=args.fp16, eval_steps=args.eval_steps, disable_tqdm=args.disable_tqdm, 
    load_best_model_at_end=args.load_best_model_at_end, metric_for_best_model=args.metric_for_best_model, greater_is_better= args.greater_is_better,)


    trainer.train()




if __name__ == "__main__":
    args = Args()
    args.init_parsearges()
    main(args)
