"""
This scripts implements the scoring logic
"""

from pathlib import Path
import logging

from tensorflow.keras.models import load_model
import pandas as pd
import click
import numpy as np

MILL_REG_MODEL_NAME = "model_1_reg.h5"
MILL_CLAS_MODEL_NAME = "model_1_clas.h5"
FURNACE_REG_MODEL_NAME = "model_2.h5"

logging.basicConfig(level=logging.INFO)

def output_mill_score(df: pd.DataFrame, output_file:str) -> None:
  """Computes mill scores and writes csv file with output
  
  Args:
      df (pd.DataFrame): Input data
      output_file (str): Output file path
  
  """
  features = ["density_error", "residue_error"]
  if any([(ft not in df.columns) for ft in features]):
    raise ValueError(f"Input data must have columns {','.join(features)}")

  regression_model = load_model(Path("models") / MILL_REG_MODEL_NAME)
  classification_model = load_model(Path("models") / MILL_CLAS_MODEL_NAME)

  def final_prediction(data: pd.DataFrame) -> np.ndarray:
    """Uses regression and classification models to calculate final mill predictions
    
    Args:
        data (pd.DataFrame): Input data
    
    Returns:
        np.ndarray: predictions
    """
    predictions = np.zeros((len(data),), dtype=np.float32)
    is_zero = classification_model.predict(data).reshape(-1)
    non_zero = is_zero <= 0.5
    predictions[non_zero] = regression_model.predict(np.array(data)[non_zero]).reshape(-1)
    return predictions

  df["water_flow_delta_pred"] = final_prediction(df[features])
  df.to_csv(output_file, index=False)
  print(df)
  logging.info(f"Wrote scored data to {output_file}.")



def output_furnace_score(df: pd.DataFrame, output_file:str) -> None:
  """Computes furnace scores and writes csv file with output
  
  Args:
      df (pd.DataFrame): Input data
      output_file (str): Output file path
  
  """
  features = ["shine_error", "measure"]
  pred_columns = ["buz_odd_delta", "buz_even_delta", "brz_odd_delta", "brz_even_delta"]

  if any([(ft not in df.columns) for ft in features]):
    raise ValueError(f"Input data must have columns {','.join(features)}")

  regression_model = load_model(Path("models") / FURNACE_REG_MODEL_NAME)
  preds_df = pd.DataFrame(regression_model.predict(df[features]), columns = [f"{col}_pred" for col in pred_columns])
  df = pd.concat([df,preds_df], axis=1)
  df.to_csv(output_file, index=False)
  print(df)
  logging.info(f"Wrote scored data to {output_file}.")


@click.command()
@click.option("--input-file", type=str, required=True, help="Input csv file to score")
@click.option("--output-file", type=str, default="output.csv", help="Output csv file path")
@click.option("--model", type=str, required=True, help="One of 'mill' or 'furnace'")
def run(
  input_file: str,
  output_file: str,
  model: str) -> None:
  """Runs scoring logic
  
  Args:
      input_file (str): Input csv file to score
      output_file (str): Output csv file path
      model (str): One of 'mill' or 'furnace'
  """

  input_df = pd.read_csv(input_file)
  logging.info(f"Read input data with {len(input_df)} rows.")

  if model == "mill":
    output_mill_score(input_df, output_file)
  elif model == "furnace":
    output_furnace_score(input_df, output_file)
  else:
    raise ValueError("model must be either 'furnace' or 'mill'")

if __name__ == "__main__":
  run()