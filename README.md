# Scoring scripts

## Set up

This project uses Python 3 and Tensorflow/Keras to run the scoring functionality. Run `pip install -r requirements.txt` from the root folder to install the dependencies

## Functionality

The `score.py` script takes 3 arguments:

1. `--input-file`: Input csv file (mandatory)
2.  `--model`: one of `mill` or `furnace` (mandatory)
3. `--output-file`: output file path (defaults to `output.csv`)

The output file will contain columns with the `_pred` suffix that will indicate the predicted attributes.

### Example

`python score.py --input-file <file> --model mill`
