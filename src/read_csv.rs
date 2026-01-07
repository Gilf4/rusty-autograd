use csv::ReaderBuilder;
use ndarray::Array2;
use std::fs::File;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CsvToArrayError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),
    #[error("Failed to parse field as f64: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),
    #[error("Inconsistent number of columns in rows")]
    InconsistentColumns,
    #[error("No data found in CSV")]
    NoData,
    #[error("Array shape error: {0}")]
    Shape(#[from] ndarray::ShapeError),
}

pub fn csv_to_array2<P: AsRef<Path>>(file_path: P) -> Result<Array2<f64>, CsvToArrayError> {
    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut data: Vec<f64> = Vec::new();
    let mut num_rows = 0;
    let mut num_cols = None;

    for result in rdr.records() {
        let record = result?;
        if record.is_empty() {
            continue;
        }
        let row: Vec<f64> = record
            .iter()
            .map(|s| s.trim().parse::<f64>())
            .collect::<Result<_, _>>()?;

        if let Some(cols) = num_cols {
            if row.len() != cols {
                return Err(CsvToArrayError::InconsistentColumns);
            }
        } else {
            num_cols = Some(row.len());
        }

        data.extend(row);
        num_rows += 1;
    }

    let cols = num_cols.unwrap_or(0);
    if num_rows == 0 || cols == 0 {
        return Err(CsvToArrayError::NoData);
    }

    Array2::from_shape_vec((num_rows, cols), data).map_err(Into::into)
}
